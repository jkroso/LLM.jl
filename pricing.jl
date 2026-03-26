@use "github.com/jkroso/Units.jl" @defunit Dimension
@use "github.com/jkroso/Units.jl/Money" USD
@use "github.com/jkroso/JSON.jl" parse_json

abstract type Tokens <: Dimension end
@defunit Token <: Tokens [k M]token

const Price = USD/Mtoken
const zero_price = (0.0USD/Mtoken, 0.0USD/Mtoken)
const API_JSON_PATH = joinpath(@__DIR__, "models.dev", "api.json")

function get_pricing(model::String)
  isfile(API_JSON_PATH) || return zero_price
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
