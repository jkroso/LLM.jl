@use "github.com/jkroso/HTTP.jl/client" parseURI send
@use "github.com/jkroso/HTTP.jl/client/Session" Session
@use "github.com/jkroso/URI.jl" URI
@use "github.com/jkroso/JSON.jl" parse_json
@use "github.com/jkroso/JSON.jl/write" JSON
@use "../abstract_provider" LLM post finalize
@use "../stream" TokenStream
@use "../pricing" Price get_pricing token

mutable struct Ollama <: LLM
  model::String
  session::Session
  uri::URI
  pricing::Tuple{Price, Price}
end

function Ollama(model::String, base_url::String)
  uri = parseURI(base_url)
  finalizer(finalize, Ollama(model, Session(uri=uri), URI("/api/chat", defaults=uri), get_pricing(model)))
end
Ollama(model::String) = Ollama(model, "http://localhost:11434")

function ollama_parse_line(s::TokenStream, line::AbstractString)
  evt = try parse_json(line) catch; return end
  msg = get(evt, "message", nothing)
  if msg !== nothing
    content = get(msg, "content", nothing)
    content !== nothing && !isempty(content) && write(s.buf, content)
  end
  if get(evt, "done", false)
    s.tokens = (token(get(evt, "prompt_eval_count", 0)), token(get(evt, "eval_count", 0)))
    s.done = true
  end
end

function (llm::Ollama)(system::String, user::String; temperature::Float64=0.7)
  payload = Dict(
    "model" => llm.model,
    "messages" => [Dict("role" => "system", "content" => system),
                   Dict("role" => "user", "content" => user)],
    "stream" => true,
    "options" => Dict("temperature" => temperature))
  req = post(llm.session, llm.uri)
  TokenStream(send(req, JSON(), payload), ollama_parse_line)
end
