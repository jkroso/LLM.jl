@use "github.com/jkroso/HTTP.jl/client" Request Header connect
@use "github.com/jkroso/HTTP.jl/client/Session" Session
@use "github.com/jkroso/URI.jl" URI
@use "github.com/jkroso/Prospects.jl/Enum" @Enum

abstract type LLM end

Base.close(llm::LLM) = close(llm.session)

finalize(llm::LLM) = try isopen(llm.session) && close(llm.session) catch end

post(s::Session, uri::URI; meta=Header()) = Request{:POST}(uri=uri, sock=connect(s), meta=meta)

# Content types

struct ImageURL
  url::String
  detail::String
end
ImageURL(url::String) = ImageURL(url, "auto")

struct ImageData
  data::Vector{UInt8}
  mime::String
  detail::String
end
ImageData(data::Vector{UInt8}, mime::String) = ImageData(data, mime, "auto")

struct Audio
  data::Vector{UInt8}
  format::String
end

const Image = Union{ImageURL, ImageData}

# Tool types

struct ToolCall
  id::String
  name::String
  arguments::Dict
end

struct Tool
  name::String
  description::String
  parameters::Dict
end

# Message types

struct SystemMessage
  text::String
end

struct UserMessage
  text::String
  images::Vector{Image}
  audio::Vector{Audio}
end
UserMessage(text::String) = UserMessage(text, Image[], Audio[])
UserMessage(text::String, images::Vector{<:Image}) = UserMessage(text, convert(Vector{Image}, images), Audio[])

struct AIMessage
  text::String
  tool_calls::Vector{ToolCall}
end
AIMessage(text::String) = AIMessage(text, ToolCall[])

struct ToolResultMessage
  tool_call_id::String
  content::String
end

const Message = Union{SystemMessage, UserMessage, AIMessage, ToolResultMessage}

# Enums

@Enum ReasoningEffort low medium high
@Enum ResponseFormat text json
