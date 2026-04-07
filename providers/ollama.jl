@use "github.com/jkroso/HTTP.jl/client" parseURI send ["Session" Session]
@use "github.com/jkroso/JSON.jl" parse_json JSON
@use "github.com/jkroso/URI.jl" URI
@use "./abstract_provider" LLM post finalize Message SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Image Tool ToolCall ReasoningEffort ResponseFormat FinishReason Document json_schema
@use "../stream" TokenStream
@use "../models" Price token
@use Base64...

mutable struct Ollama <: LLM
  info::NamedTuple
  session::Session
  uri::URI
end

function Ollama(info::NamedTuple, base_url::String)
  uri = parseURI(base_url)
  finalizer(finalize, Ollama(info, Session(uri=uri), URI("/api/chat", defaults=uri)))
end
Ollama(info::NamedTuple) = Ollama(info, "http://localhost:11434")

# Serialization

to_ollama(msg::SystemMessage) = Dict("role" => "system", "content" => msg.text)

function to_ollama(msg::UserMessage)
  !isempty(msg.audio) && error("Ollama does not support audio content")
  !isempty(msg.documents) && error("Ollama does not support document content")
  d = Dict{String,Any}("role" => "user", "content" => msg.text)
  !isempty(msg.images) && (d["images"] = [to_ollama_image(img) for img in msg.images])
  d
end

to_ollama_image(img::ImageData) = base64encode(img.data)
to_ollama_image(img::ImageURL) = error("Ollama does not support image URLs, use ImageData instead")

function to_ollama(msg::AIMessage)
  d = Dict{String,Any}("role" => "assistant", "content" => msg.text)
  if !isempty(msg.tool_calls)
    d["tool_calls"] = [Dict("function" => Dict("name" => tc.name, "arguments" => tc.arguments))
                       for tc in msg.tool_calls]
  end
  d
end

to_ollama(msg::ToolResultMessage) = Dict("role" => "tool", "content" => msg.content)

to_ollama(tool::Tool) = Dict("type" => "function",
  "function" => Dict("name" => tool.name, "description" => tool.description, "parameters" => tool.parameters))

# Stream parsing

const OLLAMA_FINISH_REASONS = Dict(
  "stop" => FinishReason.stop,
  "load" => FinishReason.stop,
  "unload" => FinishReason.stop
)

function ollama_parse_line(s::TokenStream, line::AbstractString)
  evt = try parse_json(line) catch; return end
  msg = get(evt, "message", nothing)
  if msg !== nothing
    content = get(msg, "content", nothing)
    content !== nothing && !isempty(content) && write(s.buf, content)
    thinking = get(msg, "thinking", nothing)
    thinking !== nothing && !isempty(thinking) && write(s.thinking, thinking)
    tcs = get(msg, "tool_calls", nothing)
    if tcs !== nothing
      for tc in tcs
        fn = get(tc, "function", nothing)
        fn === nothing && continue
        push!(s.tool_calls, ToolCall("", get(fn, "name", ""), get(fn, "arguments", Dict())))
      end
    end
  end
  if get(evt, "done", false)
    s.tokens = (token(get(evt, "prompt_eval_count", 0)), token(get(evt, "eval_count", 0)))
    s.finish_reason = if !isempty(s.tool_calls)
      FinishReason.tool_calls
    else
      get(OLLAMA_FINISH_REASONS, get(evt, "done_reason", "stop"), FinishReason.stop)
    end
    s.done = true
  end
end

function (llm::Ollama)(messages::Vector{<:Message};
                       temperature::Float64=0.7,
                       max_tokens::Int=8192,
                       tools::Vector{Tool}=Tool[],
                       response_format::Union{ResponseFormat,Nothing}=nothing,
                       reasoning_effort::Union{ReasoningEffort,Nothing}=nothing,
                       return_type::Union{Type,Nothing}=nothing)
  payload = Dict{String,Any}(
    "model" => llm.info.id,
    "messages" => [to_ollama(m) for m in messages],
    "stream" => true,
    "options" => Dict{String,Any}("temperature" => temperature, "num_predict" => max_tokens))
  !isempty(tools) && (payload["tools"] = [to_ollama(t) for t in tools])
  if return_type !== nothing
    payload["format"] = json_schema(return_type)
  elseif response_format !== nothing && response_format == ResponseFormat.json
    payload["format"] = "json"
  end
  reasoning_effort !== nothing && (payload["think"] = true)
  req = post(llm.session, llm.uri)
  TokenStream(send(req, JSON(), payload), ollama_parse_line)
end

(llm::Ollama)(system::String, user::String; kwargs...) =
  llm(Message[SystemMessage(system), UserMessage(user)]; kwargs...)
(llm::Ollama)(user::String; kwargs...) =
  llm(Message[UserMessage(user)]; kwargs...)
