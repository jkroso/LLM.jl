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

struct Document
  data::Vector{UInt8}
  mime::String
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
  documents::Vector{Document}
end
UserMessage(text::String) = UserMessage(text, Image[], Audio[], Document[])
UserMessage(text::String, images::Vector{<:Image}) = UserMessage(text, convert(Vector{Image}, images), Audio[], Document[])
UserMessage(text::String, images::Vector{<:Image}, audio::Vector{Audio}) = UserMessage(text, convert(Vector{Image}, images), audio, Document[])

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
@Enum FinishReason stop length tool_calls content_filter

# JSON Schema derivation

json_schema(::Type{String}) = Dict("type" => "string")
json_schema(::Type{Bool}) = Dict("type" => "boolean")
json_schema(::Type{<:Integer}) = Dict("type" => "integer")
json_schema(::Type{<:AbstractFloat}) = Dict("type" => "number")
json_schema(::Type{Vector{T}}) where T = Dict("type" => "array", "items" => json_schema(T))

function json_schema(::Type{T}) where T
  props = Dict(string(name) => json_schema(fieldtype(T, name)) for name in fieldnames(T))
  Dict("type" => "object", "properties" => props, "required" => [string.(fieldnames(T))...], "additionalProperties" => false)
end
