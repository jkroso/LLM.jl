@use "github.com/jkroso/HTTP.jl/client" Header parseURI send ["Session" Session]
@use "github.com/jkroso/JSON.jl" parse_json
@use "github.com/jkroso/JSON.jl/write" JSON
@use "github.com/jkroso/URI.jl" URI
@use "../abstract_provider" LLM post finalize Message SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Image Tool ToolCall ReasoningEffort ResponseFormat FinishReason Document
@use "../pricing" Price get_pricing token
@use "../stream" TokenStream sse
@use Base64...

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
  !isempty(msg.documents) && error("OpenAI does not support document content")
  isempty(msg.images) && isempty(msg.audio) && return Dict("role" => "user", "content" => msg.text)
  Dict("role" => "user", "content" => Any[
    Dict("type" => "text", "text" => msg.text),
    map(to_openai, msg.images)...,
    map(to_openai, msg.audio)...])
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

const OPENAI_FINISH_REASONS = Dict(
  "stop" => FinishReason.stop,
  "length" => FinishReason.length,
  "tool_calls" => FinishReason.tool_calls,
  "content_filter" => FinishReason.content_filter
)

function make_openai_parser()
  arg_bufs = Dict{Int, Tuple{String, String, IOBuffer}}() # index -> (id, name, args_buffer)

  function parse_event(s::TokenStream, data::AbstractString)
    evt = try parse_json(data) catch; return end
    choices = get(evt, "choices", nothing)
    if choices !== nothing && !isempty(choices)
      choice = choices[1]
      delta = get(choice, "delta", nothing)
      if delta !== nothing
        # Text content
        content = get(delta, "content", nothing)
        content !== nothing && write(s.buf, content)
        # Tool calls
        tcs = get(delta, "tool_calls", nothing)
        if tcs !== nothing
          for tc in tcs
            idx = get(tc, "index", 0)
            if !haskey(arg_bufs, idx)
              id = get(tc, "id", "")
              name = get(get(tc, "function", Dict()), "name", "")
              arg_bufs[idx] = (id, name, IOBuffer())
            end
            args_frag = get(get(tc, "function", Dict()), "arguments", "")
            !isempty(args_frag) && write(arg_bufs[idx][3], args_frag)
          end
        end
      end
      # Finish reason
      fr = get(choice, "finish_reason", nothing)
      if fr !== nothing
        s.finish_reason = get(OPENAI_FINISH_REASONS, fr, FinishReason.stop)
        if !isempty(arg_bufs)
          for idx in sort(collect(keys(arg_bufs)))
            id, name, buf = arg_bufs[idx]
            args_str = String(take!(buf))
            args = isempty(args_str) ? Dict() : try parse_json(args_str) catch; Dict() end
            push!(s.tool_calls, ToolCall(id, name, args))
          end
        end
      end
    end
    usage = get(evt, "usage", nothing)
    if usage !== nothing
      s.tokens = (token(get(usage, "prompt_tokens", 0)), token(get(usage, "completion_tokens", 0)))
    end
  end
  parse_event
end

function (llm::OpenAI)(messages::Vector{<:Message};
                       temperature::Float64=0.7,
                       max_tokens::Int=8192,
                       tools::Vector{Tool}=Tool[],
                       response_format::Union{ResponseFormat,Nothing}=nothing,
                       reasoning_effort::Union{ReasoningEffort,Nothing}=nothing)
  payload = Dict{String,Any}(
    "model" => llm.model,
    "messages" => [to_openai(m) for m in messages],
    "temperature" => temperature,
    "max_completion_tokens" => max_tokens,
    "stream" => true,
    "stream_options" => Dict("include_usage" => true))
  !isempty(tools) && (payload["tools"] = [to_openai(t) for t in tools])
  if response_format !== nothing
    fmt = response_format == ResponseFormat.json ? "json_object" : "text"
    payload["response_format"] = Dict("type" => fmt)
  end
  reasoning_effort !== nothing && (payload["reasoning_effort"] = string(reasoning_effort))
  req = post(llm.session, llm.uri, meta=Header("authorization" => "Bearer $(llm.api_key)"))
  TokenStream(send(req, JSON(), payload), sse(make_openai_parser()))
end

(llm::OpenAI)(system::String, user::String; kwargs...) =
  llm(Message[SystemMessage(system), UserMessage(user)]; kwargs...)
