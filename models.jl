@use "github.com/jkroso/Units.jl" @defunit Dimension ["Money" USD]
@use "github.com/jkroso/JSON.jl" parse_json write_json JSON
@use "github.com/jkroso/HTTP.jl/client" GET POST send
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

function load_providers(pids)
  missing = filter(pid -> !haskey(provider_cache, pid), pids)
  if !isempty(missing)
    all = load_cache()
    for pid in missing
      provider_cache[pid] = get(all, pid, [])
    end
  end
  vcat([provider_cache[pid] for pid in pids]...)
end

function all_models()
  result = collect(values(load_cache()))
  sort!(result, by=r->r.release_date, rev=true)
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
                max_results::Int=20)
  pq = lowercase(provider)
  mq = lowercase(model)
  ap = allowed_providers isa AbstractString ? [allowed_providers] : allowed_providers
  source = isempty(ap) ? all_models() : load_providers(ap)
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
                max_results::Int=20)
  isempty(query) && return search("", ""; allowed_providers, reasoning, vision, max_results)
  q = lowercase(query)
  ap = allowed_providers isa AbstractString ? [allowed_providers] : allowed_providers
  source = isempty(ap) ? all_models() : load_providers(ap)
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
