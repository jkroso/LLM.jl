@use "github.com/jkroso/HTTP.jl/client" Request Header parseURI connect send
@use "github.com/jkroso/HTTP.jl/client/Session" Session
@use "github.com/jkroso/URI.jl" URI
@use "github.com/jkroso/JSON.jl" parse_json
@use "github.com/jkroso/JSON.jl/write" JSON
@use "./stream" TokenStream sse
@use "./pricing" Price get_pricing token

abstract type LLM end

Base.close(llm::LLM) = close(llm.session)

finalize(llm::LLM) = isopen(llm.session) && close(llm.session)

post(s::Session, uri::URI; meta=Header()) = Request{:POST}(uri=uri, sock=connect(s), meta=meta)

# OpenAI (also used for Mistral, DeepSeek, xAI)

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

# Anthropic

mutable struct Anthropic <: LLM
  model::String
  api_key::String
  session::Session
  uri::URI
  pricing::Tuple{Price, Price}
end

function Anthropic(model::String, api_key::String)
  uri = parseURI("https://api.anthropic.com")
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

# Google

mutable struct Google <: LLM
  model::String
  api_key::String
  session::Session
  uri::URI
  pricing::Tuple{Price, Price}
end

function Google(model::String, api_key::String)
  base = parseURI("https://generativelanguage.googleapis.com")
  uri = URI("/v1beta/models/$model:streamGenerateContent?alt=sse&key=$api_key", defaults=base)
  finalizer(finalize, Google(model, api_key, Session(uri=base), uri, get_pricing(model)))
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

# Ollama

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
