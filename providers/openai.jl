@use "github.com/jkroso/HTTP.jl/client" Header parseURI send
@use "github.com/jkroso/HTTP.jl/client/Session" Session
@use "github.com/jkroso/URI.jl" URI
@use "github.com/jkroso/JSON.jl" parse_json
@use "github.com/jkroso/JSON.jl/write" JSON
@use "../abstract_provider" LLM post finalize Message SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Image Tool ToolCall ReasoningEffort ResponseFormat
using Base64
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

# Serialization

to_openai(msg::SystemMessage) = Dict("role" => "system", "content" => msg.text)

function to_openai(msg::UserMessage)
  if isempty(msg.images) && isempty(msg.audio)
    return Dict("role" => "user", "content" => msg.text)
  end
  parts = Any[Dict("type" => "text", "text" => msg.text)]
  for img in msg.images
    push!(parts, to_openai(img))
  end
  for aud in msg.audio
    push!(parts, to_openai(aud))
  end
  Dict("role" => "user", "content" => parts)
end

function to_openai(msg::AIMessage)
  d = Dict{String,Any}("role" => "assistant")
  if isempty(msg.text) && !isempty(msg.tool_calls)
    d["content"] = nothing
  else
    d["content"] = msg.text
  end
  if !isempty(msg.tool_calls)
    d["tool_calls"] = [Dict("id" => tc.id, "type" => "function",
                            "function" => Dict("name" => tc.name,
                                               "arguments" => sprint(io -> show(io, JSON(), tc.arguments))))
                       for tc in msg.tool_calls]
  end
  d
end

to_openai(msg::ToolResultMessage) = Dict("role" => "tool", "tool_call_id" => msg.tool_call_id, "content" => msg.content)

to_openai(img::ImageURL) = Dict("type" => "image_url", "image_url" => Dict("url" => img.url, "detail" => img.detail))

function to_openai(img::ImageData)
  b64 = base64encode(img.data)
  Dict("type" => "image_url", "image_url" => Dict("url" => "data:$(img.mime);base64,$b64", "detail" => img.detail))
end

to_openai(aud::Audio) = Dict("type" => "input_audio", "input_audio" => Dict("data" => base64encode(aud.data), "format" => aud.format))

to_openai(tool::Tool) = Dict("type" => "function", "function" => Dict("name" => tool.name, "description" => tool.description, "parameters" => tool.parameters))

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
