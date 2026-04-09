@use "github.com/jkroso/HTTP.jl/client" Header parseURI send ["Session" Session]
@use "github.com/jkroso/JSON.jl" parse_json JSON
@use "github.com/jkroso/URI.jl" URI
@use "./abstract_provider" LLM post finalize Message SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Tool ToolCall FinishReason
@use "../stream" TokenStream sse
@use "../models" token

mutable struct XAI <: LLM
  info::NamedTuple
  api_key::String
  session::Session
  uri::URI
  last_response_id::Union{String,Nothing}
end

function XAI(info::NamedTuple, api_key::String)
  uri = parseURI("https://api.x.ai")
  finalizer(finalize, XAI(info, api_key, Session(uri=uri), URI("/v1/responses", defaults=uri), nothing))
end

# Serialization

function to_xai(msg::UserMessage)
  isempty(msg.images) && return Dict("type" => "message", "role" => "user", "content" => msg.text)
  content = Any[Dict("type" => "input_text", "text" => msg.text)]
  for img in msg.images
    push!(content, to_xai(img))
  end
  Dict("type" => "message", "role" => "user", "content" => content)
end

function to_xai(msg::AIMessage)
  Dict("type" => "message", "role" => "assistant", "content" => msg.text)
end

to_xai(msg::ToolResultMessage) = Dict("type" => "function_call_output", "call_id" => msg.tool_call_id, "output" => msg.content)

to_xai(img::ImageURL) = Dict("type" => "input_image", "image_url" => img.url)

to_xai(tool::Tool) = Dict("type" => "function", "name" => tool.name, "description" => tool.description, "parameters" => tool.parameters)

function make_xai_parser(llm::XAI)
  function parse_event(s::TokenStream, data::AbstractString)
    evt = try parse_json(data) catch; return end
    typ = get(evt, "type", "")
    if typ == "response.created"
      resp = get(evt, "response", nothing)
      resp !== nothing && (llm.last_response_id = get(resp, "id", nothing))
    elseif typ == "response.output_text.delta"
      delta = get(evt, "delta", nothing)
      delta !== nothing && write(s.buf, delta)
    elseif typ == "response.output_item.done"
      item = get(evt, "item", nothing)
      if item !== nothing && get(item, "type", "") == "function_call"
        id = get(item, "call_id", "")
        name = get(item, "name", "")
        args_str = get(item, "arguments", "{}")
        args = try parse_json(args_str) catch; Dict() end
        push!(s.tool_calls, ToolCall(id, name, args))
      end
    elseif typ == "response.completed"
      resp = get(evt, "response", nothing)
      if resp !== nothing
        llm.last_response_id = get(resp, "id", nothing)
        usage = get(resp, "usage", nothing)
        if usage !== nothing
          s.tokens = (token(get(usage, "input_tokens", 0)), token(get(usage, "output_tokens", 0)))
        end
        status = get(resp, "status", "completed")
        if status == "incomplete"
          reason = get(resp, "incomplete_details", nothing)
          if reason !== nothing && get(reason, "reason", "") == "max_output_tokens"
            s.finish_reason = FinishReason.length
          end
        end
      end
      s.finish_reason === nothing && (s.finish_reason = FinishReason.stop)
      s.done = true
    end
  end
  parse_event
end

function (llm::XAI)(messages::Vector{<:Message};
                     temperature::Float64=0.7,
                     max_tokens::Int=8192,
                     tools::Vector=[],
                     previous_response_id::Union{String,Nothing}=llm.last_response_id)
  system_msgs = [m for m in messages if m isa SystemMessage]
  other_msgs = [m for m in messages if !(m isa SystemMessage)]

  payload = Dict{String,Any}(
    "model" => llm.info.id,
    "temperature" => temperature,
    "max_output_tokens" => max_tokens,
    "stream" => true)

  if previous_response_id !== nothing
    payload["previous_response_id"] = previous_response_id
    payload["input"] = [to_xai(other_msgs[end])]
  else
    payload["input"] = [to_xai(m) for m in other_msgs]
  end

  !isempty(system_msgs) && (payload["instructions"] = join([m.text for m in system_msgs], "\n"))

  if !isempty(tools)
    payload["tools"] = [t isa Tool ? to_xai(t) : Dict("type" => t) for t in tools]
  end

  req = post(llm.session, llm.uri, meta=Header("authorization" => "Bearer $(llm.api_key)"))
  TokenStream(send(req, JSON(), payload), sse(make_xai_parser(llm)))
end

(llm::XAI)(system::String, user::String; kwargs...) =
  llm(Message[SystemMessage(system), UserMessage(user)]; kwargs...)
(llm::XAI)(user::String; kwargs...) =
  llm(Message[UserMessage(user)]; kwargs...)
