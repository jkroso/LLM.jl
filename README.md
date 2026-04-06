# LLM.jl

LLMs are fundamentally functions of the form `llm(msg::String)::String`. This library gives you all the LLMs in basically that form. Though you will probably want to take advantage of the unique features of each LLM. It enables that too.

Model metadata (pricing, capabilities, env vars) is sourced from [models.dev](https://models.dev).

## Usage

Requires [Kip.jl](https://github.com/jkroso/Kip.jl):

```julia
@use "github.com/jkroso/LLM.jl" LLM

llm = LLM("anthropic/claude-sonnet-4-5-20250929")

# Simple call with system + user prompt
stream = llm("You are a helpful assistant", "What is Julia?")
response = read(stream, String)

# Stream token by token
stream = llm("You are a helpful assistant", "What is Julia?"; temperature=0.5)
while !eof(stream)
  print(String(readavailable(stream)))
end
```

The `provider/model` format ensures the correct provider is selected. Connections are kept alive across calls via HTTP Sessions.

## Model Search

Find models by provider, name, or capabilities:

```julia
@use "github.com/jkroso/LLM.jl/models" search

# Search by model name within a provider
search("", "claude", allowed_providers="anthropic")

# Search across specific providers
search(allowed_providers=["anthropic", "openai"])

# Filter by capabilities
search("", "", allowed_providers="openai", reasoning=true)
search("", "", allowed_providers="anthropic", vision=true)

# Single-arg OR search (matches provider or model name)
search("gpt")
```

Results are named tuples with fields: `provider`, `logo`, `env`, `id`, `name`, `release_date`, `reasoning`, `tool_call`, `modalities`, `vision`, `context`, `pricing`.

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

stream = llm(messages; tools=tools)
msg = read(stream, String)

if stream.finish_reason == FinishReason.tool_calls
  tc = stream.tool_calls[1]  # ToolCall with .id, .name, .arguments
  result = your_function(tc.arguments)

  # Send the result back
  messages = Message[messages...,
    AIMessage(msg, stream.tool_calls),
    ToolResultMessage(tc.id, result)]
  read(llm(messages), String)
end
```

## Structured Output

### JSON mode (OpenAI, Ollama)

```julia
@use "github.com/jkroso/LLM.jl" LLM ResponseFormat

stream = llm(messages; response_format=ResponseFormat.json)
result = read(stream, JSON) # parsed Dict
```

### Typed return (Anthropic, Ollama)

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

Send images via URL (OpenAI, Anthropic) or base64 data (all providers including Ollama):

```julia
@use "github.com/jkroso/LLM.jl" LLM ImageURL ImageData UserMessage

# URL (OpenAI, Anthropic)
msg = UserMessage("What's in this image?", [ImageURL("https://example.com/photo.jpg")])

# Base64 (OpenAI, Anthropic, Ollama)
img_bytes = read("photo.jpg")
msg = UserMessage("Describe this", [ImageData(img_bytes, "image/jpeg")])
```

## Documents (Anthropic only)

```julia
@use "github.com/jkroso/LLM.jl" LLM Document UserMessage Image Audio

pdf_bytes = read("document.pdf")
msg = UserMessage("Summarize this", Image[], Audio[], [Document(pdf_bytes, "application/pdf")])
```

## Reasoning

For models that support chain-of-thought (e.g. Qwen3, DeepSeek R1), the thinking trace is available on the stream:

```julia
stream = llm("What is 23 * 47?"; reasoning_effort=ReasoningEffort.high)
while !eof(stream)
  print(String(readavailable(stream)))            # final answer
  print(stderr, String(readavailable(stream.thinking))) # reasoning trace
end
```

## Token Usage & Pricing

Pricing is stored on each LLM instance from [models.dev](https://models.dev) data. Token counts are available on the stream after reading:

```julia
stream = llm("hi")
read(stream, String)

stream.tokens    # (input_tokens, output_tokens) in units of token
llm.info.pricing # (input_price, output_price) in units of USD/Mtoken
cost = sum(stream.tokens .* llm.info.pricing)
```

## API Keys

API keys are resolved from environment variables (sourced from models.dev metadata) or a config dict:

```julia
# Environment variables are auto-detected per provider (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY)
llm = LLM("openai/gpt-4o")

# Or pass explicitly via config dict
llm = LLM("openai/gpt-4o", Dict("openai_key" => "sk-..."))
```

Anthropic also supports `ANTHROPIC_BASE_URL` for custom endpoints.

## Additional Parameters

```julia
stream = llm(messages;
  temperature=0.7,           # sampling temperature (default 0.7)
  max_tokens=8192,           # max output tokens (default 8192)
  tools=Tool[],              # tool definitions
  response_format=nothing,   # ResponseFormat.json (OpenAI, Ollama)
  reasoning_effort=nothing,  # ReasoningEffort.low/medium/high
  return_type=nothing)       # Julia type for structured output (Anthropic, Ollama)
```
