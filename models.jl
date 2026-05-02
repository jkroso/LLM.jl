@use "github.com/jkroso/Units.jl" @defunit Dimension ["Money" USD]
@use "github.com/jkroso/JSON.jl" parse_json write_json JSON
@use "github.com/jkroso/HTTP.jl/client" GET POST send Header
@use Serialization: serialize, deserialize

abstract type Tokens <: Dimension end
@defunit Token <: Tokens [k M]token

const Price = USD/Mtoken
const zero_price = (0.0USD/Mtoken, 0.0USD/Mtoken)
const API_JSON_PATH = joinpath(@__DIR__, "api.json")
const CACHE_PATH = joinpath(@__DIR__, "models.jls")
const LOGOS_DIR = joinpath(@__DIR__, "logos")
const Days = 24 * 60 * 60 # in seconds

const provider_cache = Dict{String, Vector}()
const live_model_fetchers = Dict{String,Function}()
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

function provider_api_key(pid::AbstractString)
  for env in get(PROVIDER_ENVS, pid, String[])
    key = get(ENV, env, nothing)
    key !== nothing && return key
  end
  nothing
end

function fetch_openai_models(pid::AbstractString, base_url::AbstractString)
  api_key = provider_api_key(pid)
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

function parse_provider(pid, provider_data)
  logo = get_logo(get(provider_data, "logo_id", pid))
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
  data = open(parse_json, API_JSON_PATH)
  cache = Dict{String, Vector}()
  for (pid, provider_data) in data
    cache[pid] = parse_provider(pid, provider_data)
  end
  open(io->serialize(io, cache), CACHE_PATH, "w")
  cache
end

"Deserialize the provider cache, rebuilding from api.json if the file is missing or references modules that no longer load"
function load_cache()
  isfile(CACHE_PATH) || return build_cache()
  try
    deserialize(CACHE_PATH)
  catch
    build_cache()
  end
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
  try
    live = fetcher()
    results = [enrich_live_model(r, registry) for r in live]
    sort_models!(results)
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
  if !isfile(API_JSON_PATH) || (time() - mtime(API_JSON_PATH)) > 3Days
    download("https://models.dev/api.json", API_JSON_PATH)
    add_ollama_models()
  end
  if !isfile(CACHE_PATH) || mtime(CACHE_PATH) < mtime(API_JSON_PATH)
    build_cache()
  end
end

function add_ollama_models(base_url::String="http://localhost:11434")
  models = try
    read(GET("$base_url/api/tags"), String) |> parse_json
  catch
    return # Ollama not running
  end
  model_list = get(models, "models", nothing)
  model_list === nothing && return
  data = open(parse_json, API_JSON_PATH)
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
  open(API_JSON_PATH, "w") do io
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

function get_logo(provider::AbstractString)
  mkpath(LOGOS_DIR)
  path = joinpath(LOGOS_DIR, "$provider.svg")
  isfile(path) || download("https://models.dev/logos/$provider.svg", path)
  path
end

function parse_pricing(cost)
  cost === nothing && return zero_price
  input_price = get(cost, "input", nothing)
  output_price = get(cost, "output", nothing)
  (input_price === nothing || output_price === nothing) && return zero_price
  (Float64(input_price) * USD/Mtoken, Float64(output_price) * USD/Mtoken)
end
