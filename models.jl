@use "github.com/jkroso/Units.jl" @defunit Dimension ["Money" USD]
@use "github.com/jkroso/JSON.jl" parse_json write_json JSON
@use "github.com/jkroso/HTTP.jl/client" GET POST send Header
@use Serialization: serialize, deserialize

abstract type Tokens <: Dimension end
@defunit Token <: Tokens [k M]token

const Price = USD/Mtoken
const zero_price = (0.0USD/Mtoken, 0.0USD/Mtoken)
const Days = 24 * 60 * 60 # in seconds

# The folder this file was loaded from. An image built from it keeps the
# builder's path, so nothing may assume the folder exists when it runs.
const SOURCE_DIR = @__DIR__

"""
The folder that keeps the model list between runs: models.dev's `api.json`,
`models.jls` (its parse), and the logos `get_logo` downloads.

`LLM_DATA_DIR` names it. Without that it's the folder this file was loaded
from, while that folder still holds this file. An app built into an image
ships without the source, and with the builder's path for it, so there the
list goes in ~/.cache/LLM.jl. Worked out on each call, never in a constant,
for the same reason.
"""
function data_dir()
  dir = get(ENV, "LLM_DATA_DIR", "")
  isempty(dir) || return dir
  # On another user's home, stat fails with EACCES rather than saying no
  source = try isfile(joinpath(SOURCE_DIR, "models.jl")) catch; false end
  source ? SOURCE_DIR : joinpath(homedir(), ".cache", "LLM.jl")
end

api_json_path() = joinpath(data_dir(), "api.json")
cache_path() = joinpath(data_dir(), "models.jls")
logos_dir() = joinpath(data_dir(), "logos")

# The model list in use. An image built from this file keeps the one it was
# built with, so a machine that has never reached models.dev still has one.
const REGISTRY = Ref{Dict{String,Vector}}()
# The process that last refreshed REGISTRY. An image keeps the builder's, so
# every process started from one refreshes once.
const REFRESHED_IN = Ref(0)

const provider_cache = Dict{String, Vector}()
const live_model_fetchers = Dict{String,Function}()
const live_provider_cache = Dict{Tuple{String,UInt},Vector}()
const configured_fetcher_cache = Dict{Tuple{String,UInt},Function}()
const OPENAI_COMPATIBLE_URLS = Dict(
  "openai" => "https://api.openai.com",
  "mistral" => "https://api.mistral.ai",
  "deepseek" => "https://api.deepseek.com",
  "xai" => "https://api.x.ai",
)

const PROVIDER_ENVS = Dict(
  "openai" => ["OPENAI_API_KEY"],
  "mistral" => ["MISTRAL_API_KEY"],
  "deepseek" => ["DEEPSEEK_API_KEY"],
  "xai" => ["XAI_API_KEY"],
)

function parse_openai_models(provider::AbstractString, data)
  models = get(data, "data", [])
  [(provider=String(provider), id=String(m["id"]), name=String(m["id"])) for m in models if haskey(m, "id")]
end

function parse_ollama_models(data)
  models = get(data, "models", [])
  [(provider="ollama", id=String(m["name"]), name=String(m["name"])) for m in models if haskey(m, "name")]
end

function provider_api_key(pid::AbstractString, config::Dict=Dict())
  key = get(config, "$(pid)_key", nothing)
  key !== nothing && return string(key)
  for env in get(PROVIDER_ENVS, pid, String[])
    key = get(ENV, env, nothing)
    key !== nothing && return key
  end
  nothing
end

function fetch_openai_models(pid::AbstractString, base_url::AbstractString; config::Dict=Dict())
  api_key = provider_api_key(pid, config)
  api_key === nothing && error("missing API key")
  res = GET("$(base_url)/v1/models", meta=Header("authorization" => "Bearer $api_key"))
  data = parse_json(read(res, String))
  parse_openai_models(pid, data)
end

function fetch_ollama_models(base_url::AbstractString="http://localhost:11434")
  data = read(GET("$base_url/api/tags"), String) |> parse_json
  parse_ollama_models(data)
end

for (pid, url) in OPENAI_COMPATIBLE_URLS
  live_model_fetchers[pid] = () -> fetch_openai_models(pid, url)
end

live_model_fetchers["ollama"] = () -> fetch_ollama_models()

function configured_live_model_fetchers(config::Dict)
  fetchers = copy(live_model_fetchers)
  for (pid, url) in OPENAI_COMPATIBLE_URLS
    api_key = get(config, "$(pid)_key", nothing)
    api_key === nothing && continue
    cache_key = (pid, hash(string(api_key)))
    fetchers[pid] = get!(configured_fetcher_cache, cache_key) do
      () -> fetch_openai_models(pid, url; config)
    end
  end
  fetchers
end

function parse_provider(pid, provider_data)
  # The logo's id at models.dev. `get_logo` downloads it when something shows
  # it, rather than every provider's logo each time the list is parsed.
  logo = String(get(provider_data, "logo_id", pid))
  env = get(provider_data, "env", String[])
  models = get(provider_data, "models", nothing)
  models === nothing && return []
  results = map(collect(models isa Dict ? values(models) : models)) do m
    raw_mod = get(m, "modalities", nothing)
    input_mod = raw_mod !== nothing ? get(raw_mod, "input", String[]) : String[]
    output_mod = raw_mod !== nothing ? get(raw_mod, "output", String[]) : String[]
    (provider = String(pid),
     logo = logo,
     env = env,
     id = get(m, "id", ""),
     name = get(m, "name", ""),
     release_date = get(m, "release_date", ""),
     reasoning = get(m, "reasoning", false),
     tool_call = get(m, "tool_call", false),
     # Some reasoning-tier models (Anthropic Opus 4.7+, OpenAI gpt-5.x, ...)
     # reject the `temperature` parameter outright. The api.json registry
     # marks these with `"temperature": false`. Default to `true` since
     # the vast majority of models accept temperature.
     temperature = get(m, "temperature", true),
     modalities = (input=input_mod, output=output_mod),
     vision = "image" in input_mod,
     context = let l = get(m, "limit", nothing); l !== nothing ? get(l, "context", nothing) : nothing end,
     pricing = parse_pricing(get(m, "cost", nothing)))
  end
  sort!(results, by=r->r.release_date, rev=true)
end

"Rebuild the on-disk provider cache from api.json and return the new dict"
function build_cache()
  data = open(parse_json, api_json_path())
  cache = Dict{String, Vector}()
  for (pid, provider_data) in data
    cache[pid] = parse_provider(pid, provider_data)
  end
  open(io->serialize(io, cache), cache_path(), "w")
  cache
end

"Deserialize the provider cache, rebuilding from api.json if the file is missing or references modules that no longer load"
function read_cache()::Dict{String,Vector}
  isfile(cache_path()) || return build_cache()
  try
    deserialize(cache_path())
  catch
    build_cache()
  end
end

"""
The model list in use, read from disk the first time it's asked for. The
first call in a process that started from an image also refreshes it, in the
background, so the caller needn't wait for models.dev.
"""
function load_cache()
  isassigned(REGISTRY) || (REGISTRY[] = read_cache())
  if REFRESHED_IN[] != getpid() && !Base.generating_output()
    REFRESHED_IN[] = getpid()
    Threads.@spawn refresh_or_warn()
  end
  REGISTRY[]
end

function load_providers(pids; registry=load_cache(), live_fetchers=live_model_fetchers)
  isempty(pids) && return []
  vcat([provider_models(pid, registry; live_fetchers) for pid in pids]...)
end

function model_key(r)
  (String(r.provider), String(r.id))
end

function enrichment_index(registry::Dict)
  index = Dict{Tuple{String,String},Any}()
  for records in values(registry)
    for r in records
      index[model_key(r)] = r
    end
  end
  index
end

function default_model_info(provider::AbstractString, id::AbstractString; name::AbstractString=id)
  (provider=String(provider),
   logo="",
   env=String[],
   id=String(id),
   name=String(name),
   release_date="",
   reasoning=false,
   tool_call=false,
   temperature=true,
   modalities=(input=["text"], output=["text"]),
   vision=false,
   context=nothing,
   pricing=zero_price)
end

function provider_default_info(provider::AbstractString, registry::Dict)
  for r in get(registry, provider, [])
    return (env=r.env,)
  end
  (env=String[],)
end

function enrich_live_model(live, registry::Dict)
  index = enrichment_index(registry)
  name = hasproperty(live, :name) ? live.name : live.id
  defaults = provider_default_info(live.provider, registry)
  base = merge(default_model_info(live.provider, live.id; name), defaults)
  existing = get(index, model_key(base), nothing)
  existing === nothing && return base
  merge(base, existing)
end

function sort_models!(records)
  sort!(records, by=r -> isempty(r.release_date) ? "0000-00-00" : r.release_date, rev=true)
end

function provider_models(pid::AbstractString, registry::Dict=load_cache(); live_fetchers=live_model_fetchers)
  fallback = get(registry, pid, [])
  fetcher = get(live_fetchers, pid, nothing)
  fetcher === nothing && return fallback
  cache_key = (String(pid), objectid(fetcher))
  haskey(live_provider_cache, cache_key) && return live_provider_cache[cache_key]
  try
    live = fetcher()
    results = [enrich_live_model(r, registry) for r in live]
    sort_models!(results)
    live_provider_cache[cache_key] = results
    results
  catch
    fallback
  end
end

function all_models(; registry=load_cache(), live_fetchers=live_model_fetchers)
  pids = union(collect(keys(registry)), collect(keys(live_fetchers)))
  result = isempty(pids) ? [] : vcat([provider_models(pid, registry; live_fetchers) for pid in pids]...)
  sort_models!(result)
end

function __init__()
  if Base.generating_output()
    # An image build (a sysimage or pkgimage) keeps the list this loads, and
    # must not go to the network to refresh it: fetching models.dev or Ollama
    # mid-build crashes Julia 1.12 on Windows with an access violation. A stale
    # api.json bakes in fine; refreshing it is a job for run time.
    isfile(api_json_path()) || fetch_api_json()
    REGISTRY[] = read_cache()
  elseif !isassigned(REGISTRY)
    refresh_or_warn()
    REFRESHED_IN[] = getpid()
  end
  # An image starting has a list already, and refreshes it the first time
  # it's asked for one (`load_cache`), not here: a program that never asks
  # for a model makes no request.
end

"""
Download api.json again if it's missing or more than 3 days old, and parse it
again if models.jls is older than it. A list already in use is replaced by
the one on disk, which is this machine's and at most 3 days old.
"""
function refresh()
  json = api_json_path()
  (!isfile(json) || time() - mtime(json) > 3Days) && fetch_api_json()
  if !isfile(cache_path()) || mtime(cache_path()) < mtime(json)
    REGISTRY[] = build_cache()
  elseif isassigned(REGISTRY)
    REGISTRY[] = read_cache()
  end
end

# Offline, or with nowhere to write, the list in use is still a good one
function refresh_or_warn()
  try
    refresh()
  catch e
    @warn "Couldn't refresh the list of models" data_dir() exception=e
  end
end

"Download models.dev's list of models into api.json, and add the ones Ollama has here"
function fetch_api_json()
  json = api_json_path()
  mkpath(dirname(json))
  # A download cut short must not leave an api.json that looks fresh but won't parse
  tmp = json * ".part"
  try
    # Through HTTP.jl, not `download`: on Windows a libcurl download still
    # running when the program exits crashes it (EXCEPTION_ACCESS_VIOLATION in
    # Downloads' timer callback), and the refresh runs in the background.
    write(tmp, read(GET("https://models.dev/api.json"), String))
    mv(tmp, json; force=true)
  finally
    rm(tmp; force=true)
  end
  add_ollama_models()
end

function add_ollama_models(base_url::String="http://localhost:11434")
  models = try
    read(GET("$base_url/api/tags"), String) |> parse_json
  catch
    return # Ollama not running
  end
  model_list = get(models, "models", nothing)
  model_list === nothing && return
  data = open(parse_json, api_json_path())
  ollama = get!(data, "ollama") do
    Dict{String,Any}("id" => "ollama", "name" => "Ollama", "logo_id" => "ollama-cloud", "models" => Dict{String,Any}())
  end
  raw = get!(ollama, "models") do; Dict{String,Any}() end
  ollama_models = raw isa Dict ? raw : Dict{String,Any}(get(m, "id", "") => m for m in raw)
  ollama["models"] = ollama_models
  for m in model_list
    id = m["name"]
    details = get(m, "details", Dict())
    info = get_ollama_model_info(base_url, id)
    has_vision = info !== nothing && any(k->occursin(".vision.", k), keys(info))
    context = if info !== nothing
      ctx_key = findfirst(k->endswith(k, ".context_length"), keys(info))
      ctx_key !== nothing ? info[ctx_key] : nothing
    end
    input_modalities = has_vision ? ["text", "image"] : ["text"]
    ollama_models[id] = Dict{String,Any}(
      "id" => id,
      "name" => id,
      "family" => get(details, "family", ""),
      "parameter_size" => get(details, "parameter_size", ""),
      "quantization" => get(details, "quantization_level", ""),
      "open_weights" => true,
      "modalities" => Dict("input" => input_modalities, "output" => ["text"]),
      "limit" => Dict{String,Any}("context" => context))
  end
  open(api_json_path(), "w") do io
    write_json(io, data)
  end
end

function get_ollama_model_info(base_url::String, model::String)
  try
    req = POST("$base_url/api/show")
    res = send(req, JSON(), Dict("model" => model))
    data = parse(JSON(), res)
    close(req.sock)
    get(data, "model_info", nothing)
  catch
    nothing
  end
end

function matches(r; provider="", model="", reasoning=nothing, vision=nothing)
  !isempty(provider) && !occursin(provider, lowercase(r.provider)) && return false
  !isempty(model) && !occursin(model, lowercase(r.id)) && !occursin(model, lowercase(r.name)) && return false
  reasoning !== nothing && r.reasoning != reasoning && return false
  vision !== nothing && r.vision != vision && return false
  true
end

"Search for models. Filter by provider and/or model name"
function search(provider::AbstractString,
                model::AbstractString;
                allowed_providers::Union{AbstractString,AbstractVector{<:AbstractString}}=String[],
                reasoning::Union{Bool,Nothing}=nothing,
                vision::Union{Bool,Nothing}=nothing,
                max_results::Int=20,
                registry=load_cache(),
                live_fetchers=live_model_fetchers)
  pq = lowercase(provider)
  mq = lowercase(model)
  ap = allowed_providers isa AbstractString ? [allowed_providers] : allowed_providers
  source = isempty(ap) ? all_models(; registry, live_fetchers) : load_providers(ap; registry, live_fetchers)
  results = []
  for r in source
    matches(r; provider=pq, model=mq, reasoning, vision) || continue
    push!(results, r)
    length(results) >= max_results && break
  end
  results
end

"Search for models where query matches either provider or model name"
function search(query::AbstractString="";
                allowed_providers::Union{AbstractString,AbstractVector{<:AbstractString}}=String[],
                reasoning::Union{Bool,Nothing}=nothing,
                vision::Union{Bool,Nothing}=nothing,
                max_results::Int=20,
                registry=load_cache(),
                live_fetchers=live_model_fetchers)
  isempty(query) && return search("", ""; allowed_providers, reasoning, vision, max_results, registry, live_fetchers)
  q = lowercase(query)
  ap = allowed_providers isa AbstractString ? [allowed_providers] : allowed_providers
  source = isempty(ap) ? all_models(; registry, live_fetchers) : load_providers(ap; registry, live_fetchers)
  results = []
  for r in source
    occursin(q, lowercase(r.provider)) || occursin(q, lowercase(r.id)) || occursin(q, lowercase(r.name)) || continue
    reasoning !== nothing && r.reasoning != reasoning && continue
    vision !== nothing && r.vision != vision && continue
    push!(results, r)
    length(results) >= max_results && break
  end
  results
end

"The path of a model's `logo`, downloaded from models.dev the first time it's asked for"
function get_logo(provider::AbstractString)
  mkpath(logos_dir())
  path = joinpath(logos_dir(), "$provider.svg")
  isfile(path) || write(path, read(GET("https://models.dev/logos/$provider.svg"), String))
  path
end

function parse_pricing(cost)
  cost === nothing && return zero_price
  input_price = get(cost, "input", nothing)
  output_price = get(cost, "output", nothing)
  (input_price === nothing || output_price === nothing) && return zero_price
  (Float64(input_price) * USD/Mtoken, Float64(output_price) * USD/Mtoken)
end
