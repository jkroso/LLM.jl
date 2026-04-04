@use "github.com/jkroso/Units.jl" @defunit Dimension ["Money" USD]
@use "github.com/jkroso/JSON.jl" parse_json write_json JSON
@use "github.com/jkroso/HTTP.jl/client" GET POST send

abstract type Tokens <: Dimension end
@defunit Token <: Tokens [k M]token

const Price = USD/Mtoken
const zero_price = (0.0USD/Mtoken, 0.0USD/Mtoken)
const API_JSON_PATH = joinpath(@__DIR__, "api.json")
const LOGOS_DIR = joinpath(@__DIR__, "logos")
const ONE_WEEK = 7 * 24 * 60 * 60 # seconds

function __init__()
  if !isfile(API_JSON_PATH) || (time() - mtime(API_JSON_PATH)) > ONE_WEEK
    download("https://models.dev/api.json", API_JSON_PATH)
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
  data = open(parse_json, API_JSON_PATH)
  ollama = get!(data, "ollama") do
    Dict{String,Any}("id" => "ollama", "name" => "Ollama", "logo_id" => "ollama-cloud", "models" => Dict{String,Any}())
  end
  ollama_models = get!(ollama, "models") do; Dict{String,Any}() end
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

"Search for providers by name. Returns a Vector of Dicts with provider info"
function search_providers(query::AbstractVector{<:AbstractString})
  query = map(lowercase, query)
  results = Dict{String,Any}[]
  for (_, provider_data) in open(parse_json, API_JSON_PATH)
    id = lowercase(get(provider_data, "id", ""))
    name = lowercase(get(provider_data, "name", ""))
    if any(q->occursin(q, name) || occursin(q, id), query)
      push!(results, provider_data)
    end
  end
  results
end

function search_providers(query::AbstractString="")
  isempty(query) && return collect(values(open(parse_json, API_JSON_PATH)))
  search_providers(String[query])
end

"Search for models by name/id. Returns a Vector of Dicts with model info"
function search_models(query::AbstractString="";
                       provider::Union{AbstractString,AbstractVector{<:AbstractString},Nothing}=nothing,
                       reasoning::Union{Bool,Nothing}=nothing,
                       vision::Union{Bool,Nothing}=nothing,
                       max_context::Union{Int,Nothing}=nothing,
                       max_results::Int=20)
  results = Dict{String,Any}[]
  q = lowercase(query)
  for provider_data in (provider === nothing ? search_providers() : search_providers(provider))
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
        "provider" => get(provider_data, "id", ""),
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

function providers(ids::Vector{String})
  data = open(parse_json, API_JSON_PATH)
  map(ids) do id
    p = get(data, id, nothing)
    p === nothing && error("Unknown provider: $id")
    p["logo"] = get_logo(get(p, "logo_id", id))
    p
  end
end

function get_logo(provider::AbstractString)
  mkpath(LOGOS_DIR)
  path = joinpath(LOGOS_DIR, "$provider.svg")
  isfile(path) || download("https://models.dev/logos/$provider.svg", path)
  path
end

function get_pricing(model::String)
  for (_, provider_data) in open(parse_json, API_JSON_PATH)
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
