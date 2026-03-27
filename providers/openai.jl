@use "github.com/jkroso/HTTP.jl/client" Header parseURI send
@use "github.com/jkroso/HTTP.jl/client/Session" Session
@use "github.com/jkroso/URI.jl" URI
@use "github.com/jkroso/JSON.jl" parse_json
@use "github.com/jkroso/JSON.jl/write" JSON
@use "../abstract_provider" LLM post finalize
@use "../stream" TokenStream sse
@use "../pricing" Price get_pricing token

mutable struct OpenAI <: LLM
  model::String
  api_key::String
  session::Session
  uri::URI
  pricing::Tuple{Price, Price}
end

function OpenAI(model::String, api_key::String, base_url::String)
  uri = parseURI(base_url)
  finalizer(finalize, OpenAI(model, api_key, Session(uri=uri), URI("/v1/chat/completions", defaults=uri), get_pricing(model)))
end
OpenAI(model::String, api_key::String) = OpenAI(model, api_key, "https://api.openai.com")

function openai_parse_event(s::TokenStream, data::AbstractString)
  evt = try parse_json(data) catch; return end
  choices = get(evt, "choices", nothing)
  if choices !== nothing && !isempty(choices)
    delta = get(choices[1], "delta", nothing)
    if delta !== nothing
      content = get(delta, "content", nothing)
      content !== nothing && write(s.buf, content)
    end
  end
  usage = get(evt, "usage", nothing)
  if usage !== nothing
    s.tokens = (token(get(usage, "prompt_tokens", 0)), token(get(usage, "completion_tokens", 0)))
  end
end

function (llm::OpenAI)(system::String, user::String; temperature::Float64=0.7)
  payload = Dict(
    "model" => llm.model,
    "messages" => [Dict("role" => "system", "content" => system),
                   Dict("role" => "user", "content" => user)],
    "temperature" => temperature,
    "stream" => true,
    "stream_options" => Dict("include_usage" => true))
  req = post(llm.session, llm.uri, meta=Header("authorization" => "Bearer $(llm.api_key)"))
  TokenStream(send(req, JSON(), payload), sse(openai_parse_event))
end
