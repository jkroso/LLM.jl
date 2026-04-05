@use "github.com/jkroso/Units.jl" @defunit Dimension ["Money" USD]
@use "github.com/jkroso/JSON.jl" parse_json write_json JSON
@use "github.com/jkroso/HTTP.jl/client" GET POST send
@use SQLite...
@use DBInterface...

abstract type Tokens <: Dimension end
@defunit Token <: Tokens [k M]token

const Price = USD/Mtoken
const zero_price = (0.0USD/Mtoken, 0.0USD/Mtoken)
const API_JSON_PATH = joinpath(@__DIR__, "api.json")
const DB_PATH = joinpath(@__DIR__, "models.db")
const LOGOS_DIR = joinpath(@__DIR__, "logos")
const Days = 24 * 60 * 60 # in seconds

const db = Ref{SQLite.DB}()

function init_db!()
  db[] = SQLite.DB(DB_PATH)
  DBInterface.execute(db[], "DROP TABLE IF EXISTS models")
  DBInterface.execute(db[], "DROP TABLE IF EXISTS providers")
  DBInterface.execute(db[], """
    CREATE TABLE providers (
      id TEXT PRIMARY KEY,
      name TEXT,
      logo TEXT
    )""")
  DBInterface.execute(db[], """
    CREATE TABLE models (
      id TEXT,
      provider_id TEXT REFERENCES providers(id),
      name TEXT,
      release_date TEXT DEFAULT '',
      reasoning INTEGER DEFAULT 0,
      tool_call INTEGER DEFAULT 0,
      vision INTEGER DEFAULT 0,
      context INTEGER,
      input_price REAL DEFAULT 0,
      output_price REAL DEFAULT 0
    )""")
  DBInterface.execute(db[], "CREATE INDEX idx_models_release ON models(release_date DESC)")
  DBInterface.execute(db[], "CREATE INDEX idx_models_provider ON models(provider_id)")
end

function populate_db!()
  data = open(parse_json, API_JSON_PATH)
  for (_, provider_data) in data
    pid = get(provider_data, "id", "")
    existing = DBInterface.execute(db[], "SELECT 1 FROM providers WHERE id = ?", (pid,))
    if isempty(collect(existing))
      pname = get(provider_data, "name", "")
      logo = get_logo(get(provider_data, "logo_id", pid))
      DBInterface.execute(db[], "INSERT INTO providers VALUES (?, ?, ?)", (pid, pname, logo))
    end
    models = get(provider_data, "models", nothing)
    models === nothing && continue
    for model_data in (models isa Dict ? values(models) : models)
      mid = get(model_data, "id", "")
      mname = get(model_data, "name", "")
      release_date = get(model_data, "release_date", "")
      reasoning = get(model_data, "reasoning", false) ? 1 : 0
      tool_call = get(model_data, "tool_call", false) ? 1 : 0
      modalities = get(model_data, "modalities", nothing)
      vision = (modalities !== nothing && "image" in get(modalities, "input", String[])) ? 1 : 0
      limit = get(model_data, "limit", nothing)
      context = limit !== nothing ? get(limit, "context", nothing) : nothing
      cost = get(model_data, "cost", nothing)
      input_price = cost !== nothing ? get(cost, "input", 0) : 0
      output_price = cost !== nothing ? get(cost, "output", 0) : 0
      DBInterface.execute(db[], "INSERT OR IGNORE INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (mid, pid, mname, release_date, reasoning, tool_call, vision, context, input_price, output_price))
    end
  end
end

function __init__()
  if !isfile(API_JSON_PATH) || (time() - mtime(API_JSON_PATH)) > 3Days
    download("https://models.dev/api.json", API_JSON_PATH)
  end
  init_db!()
  populate_db!()
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
  logo = get_logo("ollama-cloud")
  DBInterface.execute(db[], "INSERT OR REPLACE INTO providers VALUES (?, ?, ?)", ("ollama", "Ollama", logo))
  for m in model_list
    id = m["name"]
    details = get(m, "details", Dict())
    info = get_ollama_model_info(base_url, id)
    has_vision = info !== nothing && any(k->occursin(".vision.", k), keys(info))
    context = if info !== nothing
      ctx_key = findfirst(k->endswith(k, ".context_length"), keys(info))
      ctx_key !== nothing ? info[ctx_key] : nothing
    end
    DBInterface.execute(db[], "INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      (id, "ollama", id, "", 0, 0, has_vision ? 1 : 0, context, 0, 0))
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

"Search for providers by name"
function search_providers(query::AbstractVector{<:AbstractString})
  isempty(query) && return search_providers("")
  conditions = join(["(LOWER(id) LIKE ? OR LOWER(name) LIKE ?)" for _ in query], " OR ")
  params = vcat([["%$(lowercase(q))%", "%$(lowercase(q))%"] for q in query]...)
  results = DBInterface.execute(db[], "SELECT * FROM providers WHERE $conditions", params)
  [Dict{String,Any}("id" => r.id, "name" => r.name, "logo" => r.logo) for r in results]
end

function search_providers(query::AbstractString="")
  if isempty(query)
    results = DBInterface.execute(db[], "SELECT * FROM providers")
  else
    results = DBInterface.execute(db[], "SELECT * FROM providers WHERE LOWER(id) LIKE ? OR LOWER(name) LIKE ?",
      ("%$(lowercase(query))%", "%$(lowercase(query))%"))
  end
  [Dict{String,Any}("id" => r.id, "name" => r.name, "logo" => r.logo) for r in results]
end

function make_result(r)
  Dict{String,Any}(
    "provider" => r.provider_id,
    "logo" => r.logo,
    "id" => r.id,
    "name" => r.name,
    "release_date" => something(r.release_date, ""),
    "reasoning" => r.reasoning == 1,
    "tool_call" => r.tool_call == 1,
    "modalities" => Dict("input" => r.vision == 1 ? ["text", "image"] : ["text"], "output" => ["text"]),
    "context" => r.context,
    "pricing" => parse_pricing(r.input_price, r.output_price))
end

"Search for models. Filter by provider and/or model name"
function search(provider::AbstractString,
                model::AbstractString;
                reasoning::Union{Bool,Nothing}=nothing,
                vision::Union{Bool,Nothing}=nothing,
                max_context::Union{Int,Nothing}=nothing,
                max_results::Int=20)
  conditions = String[]
  params = Any[]
  if !isempty(provider)
    push!(conditions, "(LOWER(p.id) LIKE ? OR LOWER(p.name) LIKE ?)")
    pq = "%$(lowercase(provider))%"
    push!(params, pq, pq)
  end
  if !isempty(model)
    push!(conditions, "(LOWER(m.id) LIKE ? OR LOWER(m.name) LIKE ?)")
    mq = "%$(lowercase(model))%"
    push!(params, mq, mq)
  end
  if reasoning !== nothing
    push!(conditions, "m.reasoning = ?")
    push!(params, reasoning ? 1 : 0)
  end
  if vision !== nothing
    push!(conditions, "m.vision = ?")
    push!(params, vision ? 1 : 0)
  end
  if max_context !== nothing
    push!(conditions, "m.context >= ?")
    push!(params, max_context)
  end
  where = isempty(conditions) ? "" : "WHERE " * join(conditions, " AND ")
  push!(params, max_results)
  sql = """SELECT m.*, p.logo FROM models m
           JOIN providers p ON m.provider_id = p.id
           $where ORDER BY m.release_date DESC LIMIT ?"""
  [make_result(r) for r in DBInterface.execute(db[], sql, params)]
end

"Search for models where query matches either provider or model name"
function search(query::AbstractString;
                reasoning::Union{Bool,Nothing}=nothing,
                vision::Union{Bool,Nothing}=nothing,
                max_context::Union{Int,Nothing}=nothing,
                max_results::Int=20)
  isempty(query) && return search("", ""; reasoning, vision, max_context, max_results)
  conditions = String["(LOWER(p.id) LIKE ? OR LOWER(p.name) LIKE ? OR LOWER(m.id) LIKE ? OR LOWER(m.name) LIKE ?)"]
  q = "%$(lowercase(query))%"
  params = Any[q, q, q, q]
  if reasoning !== nothing
    push!(conditions, "m.reasoning = ?")
    push!(params, reasoning ? 1 : 0)
  end
  if vision !== nothing
    push!(conditions, "m.vision = ?")
    push!(params, vision ? 1 : 0)
  end
  if max_context !== nothing
    push!(conditions, "m.context >= ?")
    push!(params, max_context)
  end
  where = "WHERE " * join(conditions, " AND ")
  push!(params, max_results)
  sql = """SELECT m.*, p.logo FROM models m
           JOIN providers p ON m.provider_id = p.id
           $where ORDER BY m.release_date DESC LIMIT ?"""
  [make_result(r) for r in DBInterface.execute(db[], sql, params)]
end

function get_logo(provider::AbstractString)
  mkpath(LOGOS_DIR)
  path = joinpath(LOGOS_DIR, "$provider.svg")
  isfile(path) || download("https://models.dev/logos/$provider.svg", path)
  path
end

function parse_pricing(input_price, output_price)
  (input_price === nothing || output_price === nothing) && return zero_price
  (Float64(input_price) * USD/Mtoken, Float64(output_price) * USD/Mtoken)
end

function get_pricing(model::String)
  results = DBInterface.execute(db[], "SELECT input_price, output_price FROM models WHERE id = ? LIMIT 1", (model,))
  for r in results
    return parse_pricing(r.input_price, r.output_price)
  end
  zero_price
end
