@use "github.com/jkroso/Units.jl" @defunit Dimension
@use "github.com/jkroso/Units.jl/Money" USD
@use "github.com/jkroso/JSON.jl" parse_json

abstract type Tokens <: Dimension end
@defunit Token <: Tokens [k M]token

const Price = USD/Mtoken
const zero_price = (0.0USD/Mtoken, 0.0USD/Mtoken)
const API_JSON_PATH = joinpath(@__DIR__, "models.dev", "api.json")
const MODELS_DEV_DIR = joinpath(@__DIR__, "models.dev")
const ONE_WEEK = 7 * 24 * 60 * 60 # seconds

function build_api_json()
  web_dir = joinpath(MODELS_DEV_DIR, "packages", "web")
  run(Cmd(`git pull`; dir=MODELS_DEV_DIR))
  run(Cmd(`bun install`; dir=MODELS_DEV_DIR))
  run(Cmd(`bun run build`; dir=web_dir))
  cp(joinpath(web_dir, "dist", "_api.json"), API_JSON_PATH; force=true)
end

is_stale(path) = (time() - mtime(path)) > ONE_WEEK

function __init__()
  if !isfile(API_JSON_PATH)
    build_api_json()
  elseif is_stale(API_JSON_PATH)
    @async build_api_json()
  end
end

load_api_data() = open(parse_json, API_JSON_PATH)

"Search for providers by name. Returns a Vector of Dicts with provider info"
function search_providers(query::AbstractString=""; max_results::Int=20)
  data = load_api_data()
  results = Dict{String,Any}[]
  q = lowercase(query)
  for (_, provider_data) in data
    id = get(provider_data, "id", "")
    name = get(provider_data, "name", "")
    if !isempty(q)
      occursin(q, lowercase(id)) || occursin(q, lowercase(name)) || continue
    end
    model_count = length(get(provider_data, "models", Dict()))
    push!(results, Dict{String,Any}(
      "id" => id,
      "name" => name,
      "doc" => get(provider_data, "doc", nothing),
      "env" => get(provider_data, "env", String[]),
      "model_count" => model_count))
  end
  sort!(results, by=r -> r["model_count"], rev=true)
  length(results) > max_results ? results[1:max_results] : results
end

"Search for models by name/id. Returns a Vector of Dicts with model info"
function search_models(query::AbstractString="";
                       provider::Union{AbstractString,AbstractVector{<:AbstractString},Nothing}=nothing,
                       reasoning::Union{Bool,Nothing}=nothing,
                       vision::Union{Bool,Nothing}=nothing,
                       max_context::Union{Int,Nothing}=nothing,
                       max_results::Int=20)
  data = load_api_data()
  results = Dict{String,Any}[]
  q = lowercase(query)
  providers = provider isa AbstractString ? [provider] : provider
  for (provider_name, provider_data) in data
    if providers !== nothing
      lp = lowercase(provider_name)
      any(p -> occursin(lowercase(p), lp), providers) || continue
    end
    models = get(provider_data, "models", nothing)
    models === nothing && continue
    for (_, model_data) in models
      id = get(model_data, "id", "")
      name = get(model_data, "name", "")
      if !isempty(q)
        occursin(q, lowercase(id)) || occursin(q, lowercase(name)) || continue
      end
      if reasoning !== nothing
        get(model_data, "reasoning", false) != reasoning && continue
      end
      if vision !== nothing
        modalities = get(model_data, "modalities", nothing)
        has_vision = modalities !== nothing && "image" in get(modalities, "input", String[])
        has_vision != vision && continue
      end
      if max_context !== nothing
        limit = get(model_data, "limit", nothing)
        ctx = limit !== nothing ? get(limit, "context", 0) : 0
        ctx < max_context && continue
      end
      push!(results, Dict{String,Any}(
        "provider" => provider_name,
        "id" => id,
        "name" => name,
        "release_date" => get(model_data, "release_date", ""),
        "reasoning" => get(model_data, "reasoning", false),
        "tool_call" => get(model_data, "tool_call", false),
        "modalities" => get(model_data, "modalities", nothing),
        "context" => let l = get(model_data, "limit", nothing); l !== nothing ? get(l, "context", nothing) : nothing end,
        "cost" => get(model_data, "cost", nothing)))
    end
  end
  sort!(results, by=r -> r["release_date"], rev=true)
  length(results) > max_results ? results[1:max_results] : results
end

function get_pricing(model::String)
  data = load_api_data()
  for (_, provider_data) in data
    models = get(provider_data, "models", nothing)
    models === nothing && continue
    for (_, model_data) in models
      get(model_data, "id", nothing) == model || continue
      cost = get(model_data, "cost", nothing)
      cost === nothing && return zero_price
      input_price = get(cost, "input", nothing)
      output_price = get(cost, "output", nothing)
      (input_price === nothing || output_price === nothing) && return zero_price
      return (Float64(input_price) * USD/Mtoken, Float64(output_price) * USD/Mtoken)
    end
  end
  zero_price
end
