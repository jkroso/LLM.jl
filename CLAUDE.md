# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM.jl is a Julia library providing a unified streaming interface to multiple LLM providers (OpenAI, Anthropic, Google, Ollama, plus OpenAI-compatible services like Mistral, DeepSeek, xAI). It uses Kip's `@use` macro system for imports (not standard `using`/`import`).

## Module System

This project uses `@use` (from github.com/jkroso/Kip.jl) instead of Julia's standard module system. There is no `Project.toml` -- dependencies are resolved by URL path (e.g., `@use "github.com/jkroso/HTTP.jl/client"`). Relative imports like `@use "./messages"` import from sibling files. There is no `module` declaration; each file is implicitly its own module.

## Architecture

- **`main.jl`** -- Entry point. Exports `LLM(model_name, config)` constructor which dispatches model name strings to provider constructors. Model routing is prefix-based (e.g., `"claude*"` -> Anthropic, `"gpt*"` -> OpenAI). Unknown models default to Ollama.
- **`messages.jl`** -- Simple message types: `SystemMessage`, `UserMessage`, `AIMessage`, unified as `AbstractMessage = Union{...}`.
- **`stream.jl`** -- `TokenStream <: IO`, a mutable struct that wraps an HTTP response and parses streaming lines into a readable stream. The `sse()` wrapper handles SSE protocol (data: prefix, [DONE] sentinel) for cloud providers. Ollama uses NDJSON and passes its parser directly. Implements `Base.read`, `readavailable`, `eof`, etc.
- **`models.jl`** -- Defines `Token` unit via Units.jl and `Price = USD/Mtoken`. `get_pricing(model)` reads `models.dev/api.json` on demand and returns `(input_price, output_price)` tuple, defaulting to zero. `search_models(query; provider, reasoning, vision, max_context, max_results)` searches the model database.
- **`providers.jl`** -- All providers in one file. Each is a `mutable struct <: LLM` with `model`, `session` (HTTP Session for keep-alive), `uri` (pre-built request URI), and `pricing`. Each struct is callable, returning a `TokenStream`. Providers have finalizers to close sessions on GC, plus `Base.close(::LLM)` for explicit cleanup.

## Provider Call Pattern

Each provider struct is callable: `llm(messages; temperature=0.7)` -> `TokenStream`. The TokenStream implements Julia's IO interface so callers can `read(stream, String)` for the full response or `readavailable(stream)` for incremental streaming. Token usage is stored in `stream.meta` (keys: `"input_tokens"`, `"output_tokens"`).

## Running Tests

```
julia test.jl
```

Or from the Kaimon REPL: `cd("/path/to/LLM.jl"); include("test.jl")`

## models.dev Submodule

`models.dev/` is a git submodule (github.com/sst/models.dev) -- a TypeScript/Bun project that provides an open-source database of AI model specs and pricing. The Julia code only reads `models.dev/api.json` for pricing data. To generate it:

```
cd models.dev && bun install && cd packages/web && bun run build
cp packages/web/dist/_api.json ../api.json
```

## Key Dependencies

- `github.com/jkroso/HTTP.jl/client` -- HTTP client with Session keep-alive support
- `github.com/jkroso/JSON.jl` -- JSON parsing (`parse_json`) and serialization (`JSON` MIME type)
- `github.com/jkroso/Units.jl` -- Unit types (`Token`, `Price = USD/Mtoken`)
- `github.com/jkroso/URI.jl` -- URI type used for base URLs and request paths
