@use "github.com/jkroso/HTTP.jl/client" Header parseURI send
@use "github.com/jkroso/HTTP.jl/client/Session" Session
@use "github.com/jkroso/JSON.jl" parse_json
@use "github.com/jkroso/JSON.jl/write" JSON
@use "github.com/jkroso/URI.jl" URI
@use "../abstract_provider" LLM post finalize Message SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Image Tool ToolCall ReasoningEffort ResponseFormat FinishReason Document json_schema
@use "../pricing" Price get_pricing token
@use "../stream" TokenStream sse

using Base64

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

# Serialization

function to_anthropic(msg::UserMessage)
  !isempty(msg.audio) && error("Anthropic does not support audio content")
  has_media = !isempty(msg.images) || !isempty(msg.documents)
  has_media || return Dict("role" => "user", "content" => msg.text)
  parts = Any[Dict("type" => "text", "text" => msg.text)]
  for img in msg.images
    push!(parts, to_anthropic(img))
  end
  for doc in msg.documents
    push!(parts, to_anthropic(doc))
  end
  Dict("role" => "user", "content" => parts)
end

function to_anthropic(msg::AIMessage)
  isempty(msg.tool_calls) && return Dict("role" => "assistant", "content" => msg.text)
  parts = Any[]
  !isempty(msg.text) && push!(parts, Dict("type" => "text", "text" => msg.text))
  for tc in msg.tool_calls
    push!(parts, Dict("type" => "tool_use", "id" => tc.id, "name" => tc.name, "input" => tc.arguments))
  end
  Dict("role" => "assistant", "content" => parts)
end

to_anthropic(msg::ToolResultMessage) = Dict("role" => "user", "content" => [
  Dict("type" => "tool_result", "tool_use_id" => msg.tool_call_id, "content" => msg.content)])

to_anthropic(img::ImageURL) = Dict("type" => "image", "source" => Dict("type" => "url", "url" => img.url))

function to_anthropic(img::ImageData)
  Dict("type" => "image", "source" => Dict("type" => "base64", "media_type" => img.mime, "data" => base64encode(img.data)))
end

function to_anthropic(doc::Document)
  Dict("type" => "document", "source" => Dict("type" => "base64", "media_type" => doc.mime, "data" => base64encode(doc.data)))
end

to_anthropic(tool::Tool) = Dict("name" => tool.name, "description" => tool.description, "input_schema" => tool.parameters)

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
