@use "github.com/jkroso/HTTP.jl/client" Response
@use "./pricing" token

mutable struct TokenStream <: IO
  response::Response
  parse_line::Function
  buf::IOBuffer
  tokens::Tuple{token,token}
  done::Bool
  leftover::String
end

TokenStream(response::Response, parse_line::Function) =
  TokenStream(response, parse_line, PipeBuffer(), (token(0), token(0)), false, "")

"Wrap a parse_event function to handle SSE protocol (data: prefix, [DONE] sentinel)"
sse(parse_event::Function) = (s::TokenStream, line::AbstractString) -> begin
  startswith(line, "data: ") || return
  data = line[7:end]
  data == "[DONE]" && (s.done = true; return)
  parse_event(s, data)
end

function Base.eof(s::TokenStream)
  if s.done && bytesavailable(s.buf) == 0
    # Drain remaining HTTP response so the session socket is clean for reuse
    try while !eof(s.response) readavailable(s.response) end catch end
    return true
  end
  false
end
Base.isopen(s::TokenStream) = !s.done
Base.bytesavailable(s::TokenStream) = bytesavailable(s.buf)

function Base.readavailable(s::TokenStream)
  if bytesavailable(s.buf) > 0
    return readavailable(s.buf)
  end
  pull_tokens!(s)
  readavailable(s.buf)
end

function Base.read(s::TokenStream, ::Type{UInt8})
  while bytesavailable(s.buf) == 0 && !s.done
    pull_tokens!(s)
  end
  read(s.buf, UInt8)
end

function Base.read(s::TokenStream)
  chunks = UInt8[]
  while !eof(s)
    append!(chunks, readavailable(s))
  end
  chunks
end

function Base.close(s::TokenStream)
  s.done = true
  # Drain remaining HTTP response so the session socket is clean for reuse
  try while !eof(s.response) readavailable(s.response) end catch end
  nothing
end

"Read the full response as a String"
Base.read(s::TokenStream, ::Type{String}) = String(read(s))

"Read available data from response, parse lines, write text to buffer"
function pull_tokens!(s::TokenStream)
  s.done && return
  eof(s.response) && (s.done = true; return)
  raw = readavailable(s.response)
  isempty(raw) && return
  text = s.leftover * String(raw)
  lines = split(text, '\n')
  s.leftover = String(lines[end])
  for i in 1:length(lines)-1
    line = rstrip(lines[i], '\r')
    isempty(line) && continue
    s.parse_line(s, line)
  end
end
