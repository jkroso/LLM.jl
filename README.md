# LLM.jl

LLMs are fundamentally functions of the form `llm(msg::String)::String`. This library gives you all the LLMs in basically that form. Though you will probably want to take advantage of the unique features of each LLM. It enables that too.

## Supported Providers

| Provider | Model prefixes | API |
|----------|---------------|-----|
| xAI | `grok*` | OpenAI-compatible |
| OpenAI | `gpt*`, `o1*`, `o3*`, `o4*` | OpenAI |
| Anthropic | `claude*`, `anthropic*` | Anthropic |
| Google | `gemini*` | Gemini |
| Mistral | `mistral*` | OpenAI-compatible |
| DeepSeek | `deepseek*` | OpenAI-compatible |
| Ollama | `ollama:*` or any unrecognized model | Local |

## Usage

Requires [Kip.jl](https://github.com/jkroso/Kip.jl):

```julia
@use "github.com/jkroso/LLM.jl" LLM

llm = LLM("claude-sonnet-4-5-20250929")

# Simple call with system + user prompt
stream = llm("You are a helpful assistant", "What is Julia?")
response = read(stream, String)

# Stream token by token
stream = llm("You are a helpful assistant", "What is Julia?"; temperature=0.5)
while !eof(stream)
  print(String(readavailable(stream)))
end
```

Connections are kept alive across calls via HTTP Sessions, so repeated requests to the same provider reuse the TCP connection.

## Multi-turn Conversations

Pass a `Vector{Message}` for multi-turn conversations:

```julia
@use "github.com/jkroso/LLM.jl" LLM Message SystemMessage UserMessage AIMessage ToolResultMessage

messages = Message[
  SystemMessage("You are a helpful assistant"),
  UserMessage("My name is Alice"),
  AIMessage("Nice to meet you, Alice!"),
  UserMessage("What is my name?")
]

read(llm(messages), String) # "Your name is Alice!"
```

## Tool Calling

To offer up tools in the LLM's preferred format you can pass a tools parameter:

```julia
@use "github.com/jkroso/LLM.jl" LLM Tool ToolCall ToolResultMessage FinishReason Message SystemMessage UserMessage

tools = [Tool("get_weather", "Get the current weather", Dict(
  "type" => "object",
  "properties" => Dict("city" => Dict("type" => "string")),
  "required" => ["city"]))]

messages = Message[
  SystemMessage("Use tools to answer questions"),
  UserMessage("What's the weather in Boston?")
]

res = llm(messages; tools=tools)
msg = read(res, String)

if res.finish_reason == FinishReason.tool_calls
  tc = res.tool_calls[1]  # ToolCall with .id, .name, .arguments
  result = your_function(tc.arguments)

  # Send the result back
  messages = Message[messages...,
    AIMessage(msg, res.tool_calls),
    ToolResultMessage(tc.id, result)]
  read(llm(messages), String)
end
```

## Structured Output

### OpenAI (JSON mode)

```julia
@use "github.com/jkroso/LLM.jl" LLM ResponseFormat

stream = llm(messages; response_format=ResponseFormat.json)
result = read(stream, JSON) # parsed Dict
```

### Anthropic (typed return)

Derive a JSON schema from a Julia struct and get back a typed result:

```julia
@use "github.com/jkroso/LLM.jl" LLM from_json

struct Person
  name::String
  age::Int
end

stream = llm("Make up a person"; return_type=Person)
result = from_json(stream, Person) # Person("Jeb", 21)
```

## Images

Send images via URL or base64 data:

```julia
@use "github.com/jkroso/LLM.jl" LLM ImageURL ImageData UserMessage

# URL
msg = UserMessage("What's in this image?", [ImageURL("https://example.com/photo.jpg")])

# Base64
img_bytes = read("photo.jpg")
msg = UserMessage("Describe this", [ImageData(img_bytes, "image/jpeg")])
```

## Documents (Anthropic only)

```julia
@use "github.com/jkroso/LLM.jl" LLM Document UserMessage Image Audio

pdf_bytes = read("document.pdf")
msg = UserMessage("Summarize this", Image[], Audio[], [Document(pdf_bytes, "application/pdf")])
```

## Finish Reason

Check why the model stopped generating:

```julia
@use "github.com/jkroso/LLM.jl" FinishReason

stream.finish_reason == FinishReason.stop        # natural completion
stream.finish_reason == FinishReason.length       # hit max_tokens
stream.finish_reason == FinishReason.tool_calls   # wants to call tools
stream.finish_reason == FinishReason.content_filter # content filtered
```

Values are normalized across providers.

## Additional Parameters

```julia
stream = llm(messages;
  temperature=0.7,           # sampling temperature (default 0.7)
  max_tokens=8192,            # max output tokens (default 8192)
  tools=Tool[],               # tool definitions
  response_format=nothing,    # ResponseFormat.json (OpenAI only)
  reasoning_effort=nothing,   # ReasoningEffort.low/medium/high
  return_type=nothing)        # Julia type for structured output (Anthropic only)
```

## Token Usage & Pricing

Pricing is cached on each LLM instance at construction from [models.dev](https://models.dev) data. Token counts are available on the stream after reading, so you can compute cost:

```julia
stream = llm("hi")
read(stream, String) # must be read fully before tokens and pricing become available

stream.tokens # a tuple (input_tokens, output_tokens) in units of token
llm.pricing   # a tuple (input_price, output_price) in units of USD/Mtoken
cost = sum(stream.tokens .* llm.pricing) # e.g. 0.02 USD
```

## API Keys

Set via environment variables or pass in a config dict:

| Provider | Env variable |
|----------|-------------|
| xAI | `XAI_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |

Anthropic also supports `ANTHROPIC_BASE_URL` for custom endpoints.

```julia
# Using config dict
llm = LLM("gpt-4", Dict("openai_key" => "sk-..."))
```

## Direct Provider Construction

You can also construct providers directly instead of using `LLM`:

```julia
@use "github.com/jkroso/LLM.jl/providers" OpenAI Anthropic Google Ollama

llm = Anthropic("claude-sonnet-4-5-20250929", ENV["ANTHROPIC_API_KEY"])
llm = OpenAI("gpt-4", ENV["OPENAI_API_KEY"])
llm = Google("gemini-pro", ENV["GOOGLE_API_KEY"])
llm = Ollama("llama3")
llm = Ollama("llama3", "http://remote-host:11434")

# OpenAI-compatible providers
llm = OpenAI("mistral-large", ENV["MISTRAL_API_KEY"], "https://api.mistral.ai")

# Clean up when done or let GC do it
close(llm)
```
