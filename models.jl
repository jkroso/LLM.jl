@use "github.com/jkroso/Units.jl" @defunit Dimension ["Money" USD]
@use "github.com/jkroso/Promises.jl/ephemeral" @ephemeral need
@use "github.com/jkroso/JSON.jl" parse_json write_json JSON
@use "github.com/jkroso/HTTP.jl/client" GET POST send
@use Dates: Second

abstract type Tokens <: Dimension end
@defunit Token <: Tokens [k M]token

const Price = USD/Mtoken
const zero_price = (0.0USD/Mtoken, 0.0USD/Mtoken)
const API_JSON_PATH = joinpath(@__DIR__, "api.json")
const LOGOS_DIR = joinpath(@__DIR__, "logos")
const Days = 24 * 60 * 60 # in seconds

const db = @ephemeral Second(15) open(parse_json, API_JSON_PATH)

function __init__()
  if !isfile(API_JSON_PATH) || (time() - mtime(API_JSON_PATH)) > 3Days
    download("https://models.dev/api.json", API_JSON_PATH)
    sort_models!()
  end
  add_ollama_models()
end

"Build a flat globally-sorted model index and save to disk"
function sort_models!()
  data = open(parse_json, API_JSON_PATH)
  sorted = Any[]
  for (key, provider_data) in data
    key == "_index" && continue
    pid = get(provider_data, "id", "")
    logo_id = get(provider_data, "logo_id", pid)
    models = get(provider_data, "models", nothing)
    models === nothing && continue
    for m in (models isa Dict ? values(models) : models)
      push!(sorted, Dict{String,Any}("provider" => pid, "logo_id" => logo_id, "model" => m))
    end
  end
  sort!(sorted, by=e->get(e["model"], "release_date", ""), rev=true)
  data["_index"] = sorted
  open(API_JSON_PATH, "w") do io
    write_json(io, data)
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
  sort_models!()
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

"Search for providers by name"
function search_providers(query::AbstractVector{<:AbstractString})
  isempty(query) && return search_providers("")
  query = map(lowercase, query)
  results = Dict{String,Any}[]
  for (key, provider_data) in need(db)
    key == "_index" && continue
    id = lowercase(get(provider_data, "id", ""))
    name = lowercase(get(provider_data, "name", ""))
    if any(q->occursin(q, name) || occursin(q, id), query)
      push!(results, provider_data)
    end
  end
  results
end

function search_providers(query::AbstractString="")
  isempty(query) && return [v for (k, v) in need(db) if k != "_index"]
  search_providers(String[query])
end

function make_result(entry)
  m = entry["model"]
  Dict{String,Any}(
    "provider" => get(entry, "provider", ""),
    "logo" => get_logo(get(entry, "logo_id", get(entry, "provider", ""))),
    "id" => get(m, "id", ""),
    "name" => get(m, "name", ""),
    "release_date" => get(m, "release_date", ""),
    "reasoning" => get(m, "reasoning", false),
    "tool_call" => get(m, "tool_call", false),
    "modalities" => get(m, "modalities", nothing),
    "context" => let l = get(m, "limit", nothing); l !== nothing ? get(l, "context", nothing) : nothing end,
    "pricing" => parse_pricing(get(m, "cost", nothing)))
end

function matches_filters(model_data; reasoning=nothing, vision=nothing, max_context=nothing)
  if reasoning !== nothing
    get(model_data, "reasoning", false) != reasoning && return false
  end
  if vision !== nothing
    modalities = get(model_data, "modalities", nothing)
    has_vision = modalities !== nothing && "image" in get(modalities, "input", String[])
    has_vision != vision && return false
  end
  if max_context !== nothing
    limit = get(model_data, "limit", nothing)
    ctx = limit !== nothing ? get(limit, "context", 0) : 0
    ctx < max_context && return false
  end
  true
end

"Search for models. Filter by provider and/or model name"
function search(provider::AbstractString,
                model::AbstractString;
                allowed_providers::Union{AbstractString,AbstractVector{<:AbstractString},Nothing}=nothing,
                reasoning::Union{Bool,Nothing}=nothing,
                vision::Union{Bool,Nothing}=nothing,
                max_context::Union{Int,Nothing}=nothing,
                max_results::Int=20)
  results = Dict{String,Any}[]
  pq = lowercase(provider)
  mq = lowercase(model)
  ap = allowed_providers isa AbstractString ? [allowed_providers] : allowed_providers
  for entry in get(need(db), "_index", [])
    pid = get(entry, "provider", "")
    ap !== nothing && !(pid in ap) && continue
    if !isempty(pq)
      occursin(pq, lowercase(pid)) || continue
    end
    m = entry["model"]
    if !isempty(mq)
      mid = lowercase(get(m, "id", ""))
      mname = lowercase(get(m, "name", ""))
      occursin(mq, mid) || occursin(mq, mname) || continue
    end
    matches_filters(m; reasoning, vision, max_context) || continue
    push!(results, make_result(entry))
    length(results) >= max_results && break
  end
  results
end

"Search for models where query matches either provider or model name"
function search(query::AbstractString="";
                allowed_providers::Union{AbstractString,AbstractVector{<:AbstractString},Nothing}=nothing,
                reasoning::Union{Bool,Nothing}=nothing,
                vision::Union{Bool,Nothing}=nothing,
                max_context::Union{Int,Nothing}=nothing,
                max_results::Int=20)
  isempty(query) && return search("", ""; allowed_providers, reasoning, vision, max_context, max_results)
  q = lowercase(query)
  ap = allowed_providers isa AbstractString ? [allowed_providers] : allowed_providers
  results = Dict{String,Any}[]
  for entry in get(need(db), "_index", [])
    pid = get(entry, "provider", "")
    ap !== nothing && !(pid in ap) && continue
    m = entry["model"]
    mid = lowercase(get(m, "id", ""))
    mname = lowercase(get(m, "name", ""))
    occursin(q, lowercase(pid)) || occursin(q, mid) || occursin(q, mname) || continue
    matches_filters(m; reasoning, vision, max_context) || continue
    push!(results, make_result(entry))
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

function get_pricing(model::String)
  for (key, provider_data) in need(db)
    key == "_index" && continue
    models = get(provider_data, "models", nothing)
    models === nothing && continue
    for model_data in (models isa Dict ? values(models) : models)
      get(model_data, "id", nothing) == model || continue
      return parse_pricing(get(model_data, "cost", nothing))
    end
  end
  zero_price
end
