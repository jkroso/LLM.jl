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

function get_pricing(model::String)
  data = open(parse_json, API_JSON_PATH)
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
