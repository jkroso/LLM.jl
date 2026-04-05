@use "./providers/abstract_provider" LLM Message SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Image Tool ToolCall ReasoningEffort ResponseFormat FinishReason Document json_schema
@use "./providers" OpenAI Anthropic Google Ollama
@use "./stream" from_json
@use "./models" get_pricing search

const PROVIDER_MAP = Dict(
  "anthropic" => (key="anthropic_key", url=nothing,                    env=["ANTHROPIC_API_KEY"]),
  "openai"    => (key="openai_key",    url=nothing,                    env=["OPENAI_API_KEY"]),
  "google"    => (key="google_key",    url=nothing,                    env=["GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"]),
  "mistral"   => (key="mistral_key",   url="https://api.mistral.ai",   env=["MISTRAL_API_KEY"]),
  "deepseek"  => (key="deepseek_key",  url="https://api.deepseek.com", env=["DEEPSEEK_API_KEY"]),
  "xai"       => (key="xai_key",       url="https://api.x.ai",         env=["XAI_API_KEY"]),
)

"Look up the API key for a provider from config dict or environment variables"
function get_api_key(provider_id::String, config::Dict)
  info = PROVIDER_MAP[provider_id]
  key = get(config, info.key, nothing)
  key !== nothing && return string(key)
  for env in info.env
    val = get(ENV, env, nothing)
    val !== nothing && return val
  end
  ""
end

"Create an LLM instance from a model name string and config dict"
function LLM(model::String, config::Dict=Dict())
  lm = lowercase(model)
  # Handle provider-prefixed IDs like "xai/grok-4-fast" or "anthropic/claude-sonnet-4-6"
  if contains(lm, '/')
    prefix, name = split(lm, '/'; limit=2)
    if prefix == "ollama"
      return Ollama(String(name), get(config, "ollama_url", "http://localhost:11434"))
    end
    info = get(PROVIDER_MAP, prefix, nothing)
    if info !== nothing
      api_key = get_api_key(prefix, config)
      if prefix == "anthropic"
        return Anthropic(model, api_key)
      elseif prefix == "google"
        return Google(model, api_key)
      elseif info.url !== nothing
        return OpenAI(model, api_key, info.url; provider=prefix)
      else
        return OpenAI(model, api_key; provider=prefix)
      end
    end
  end
  # Fallback: detect by model name prefix
  if startswith(lm, "ollama:")
    Ollama(model[length("ollama:")+1:end], get(config, "ollama_url", "http://localhost:11434"))
  elseif startswith(lm, "claude") || startswith(lm, "anthropic")
    Anthropic(model, get_api_key("anthropic", config))
  elseif startswith(lm, "gpt") || startswith(lm, "o1") || startswith(lm, "o3") || startswith(lm, "o4")
    OpenAI(model, get_api_key("openai", config))
  elseif startswith(lm, "gemini")
    Google(model, get_api_key("google", config))
  elseif startswith(lm, "mistral")
    OpenAI(model, get_api_key("mistral", config), "https://api.mistral.ai"; provider="mistral")
  elseif startswith(lm, "deepseek")
    OpenAI(model, get_api_key("deepseek", config), "https://api.deepseek.com"; provider="deepseek")
  elseif startswith(lm, "grok")
    OpenAI(model, get_api_key("xai", config), "https://api.x.ai"; provider="xai")
  else
    Ollama(model, get(config, "ollama_url", "http://localhost:11434"))
  end
end
