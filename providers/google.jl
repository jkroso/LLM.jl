@use "github.com/jkroso/HTTP.jl/client" parseURI send ["Session" Session]
@use "github.com/jkroso/JSON.jl" parse_json JSON
@use "github.com/jkroso/URI.jl" URI
@use "./abstract_provider" LLM post finalize
@use "../stream" TokenStream sse
@use "../models" Price token

mutable struct Google <: LLM
  info::NamedTuple
  api_key::String
  session::Session
  uri::URI
end

function Google(info::NamedTuple, api_key::String)
  base = parseURI("https://generativelanguage.googleapis.com")
  uri = URI("/v1beta/models/$(info.id):streamGenerateContent?alt=sse&key=$api_key", defaults=base)
  finalizer(finalize, Google(info, api_key, Session(uri=base), uri))
end

function google_parse_event(s::TokenStream, data::AbstractString)
  evt = try parse_json(data) catch; return end
  candidates = get(evt, "candidates", nothing)
  if candidates !== nothing && !isempty(candidates)
    content = get(candidates[1], "content", nothing)
    if content !== nothing
      parts = get(content, "parts", nothing)
      if parts !== nothing && !isempty(parts)
        text = get(parts[1], "text", nothing)
        text !== nothing && write(s.buf, text)
      end
    end
  end
  usage = get(evt, "usageMetadata", nothing)
  if usage !== nothing
    s.tokens = (token(get(usage, "promptTokenCount", 0)), token(get(usage, "candidatesTokenCount", 0)))
  end
end

function (llm::Google)(system::String, user::String; temperature::Float64=0.7)
  payload = Dict{String,Any}(
    "contents" => [Dict("role" => "user", "parts" => [Dict("text" => user)])],
    "generationConfig" => Dict("temperature" => temperature))
  !isempty(system) && (payload["systemInstruction"] = Dict("parts" => [Dict("text" => system)]))
  req = post(llm.session, llm.uri)
  TokenStream(send(req, JSON(), payload), sse(google_parse_event))
end
