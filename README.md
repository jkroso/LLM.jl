# LLM.jl

LLMs are fundamentally functions of the form `llm(msg::String)::String`. This library gives you all the LLMs in basically that form.

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

stream = llm("You are a helpful assistant", "What is Julia?") # call the LLM with a system prompt and a user prompt
response = read(stream, String) # Read full response

# Stream token by token
stream = llm(system_prompt, user_prompt, temperature=0.5)
while !eof(stream)
  print(String(readavailable(stream)))
end
```

Connections are kept alive across calls via HTTP Sessions, so repeated requests to the same provider reuse the TCP connection.

## Token Usage & Pricing

Pricing is cached on each LLM instance at construction from [models.dev](https://models.dev) data. Token counts are available on the stream after reading, so you can compute cost:

```julia
stream = llm(system_prompt, user_prompt)
read(stream, String)

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