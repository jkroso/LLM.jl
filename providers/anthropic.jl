@use "github.com/jkroso/HTTP.jl/client" Header parseURI send
@use "github.com/jkroso/HTTP.jl/client/Session" Session
@use "github.com/jkroso/JSON.jl" parse_json
@use "github.com/jkroso/JSON.jl/write" JSON
@use "github.com/jkroso/URI.jl" URI
@use "../abstract_provider" LLM post finalize
@use "../pricing" Price get_pricing token
@use "../stream" TokenStream sse

mutable struct Anthropic <: LLM
  model::String
  api_key::String
  session::Session
  uri::URI
  pricing::Tuple{Price, Price}
end

function Anthropic(model::String, api_key::String)
  uri = parseURI(get(ENV, "ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
  finalizer(finalize, Anthropic(model, api_key, Session(uri=uri), URI("/v1/messages", defaults=uri), get_pricing(model)))
end

function anthropic_parse_event(s::TokenStream, data::AbstractString)
  evt = try parse_json(data) catch; return end
  typ = get(evt, "type", "")
  if typ == "content_block_delta"
    delta = get(evt, "delta", nothing)
    if delta !== nothing
      text = get(delta, "text", nothing)
      text !== nothing && write(s.buf, text)
    end
  elseif typ == "message_delta"
    usage = get(evt, "usage", nothing)
    if usage !== nothing
      s.tokens = (s.tokens[1], token(get(usage, "output_tokens", 0)))
    end
  elseif typ == "message_start"
    msg = get(evt, "message", nothing)
    if msg !== nothing
      usage = get(msg, "usage", nothing)
      if usage !== nothing
        s.tokens = (token(get(usage, "input_tokens", 0)), s.tokens[2])
      end
    end
  elseif typ == "message_stop"
    s.done = true
  end
end

function (llm::Anthropic)(system::String, user::String; temperature::Float64=0.7, max_tokens::Int=8192)
  payload = Dict{String,Any}(
    "model" => llm.model,
    "messages" => [Dict("role" => "user", "content" => user)],
    "temperature" => temperature,
    "max_tokens" => max_tokens,
    "stream" => true)
  !isempty(system) && (payload["system"] = system)
  req = post(llm.session, llm.uri, meta=Header(
    "x-api-key" => llm.api_key,
    "anthropic-version" => "2023-06-01"))
  TokenStream(send(req, JSON(), payload), sse(anthropic_parse_event))
end
