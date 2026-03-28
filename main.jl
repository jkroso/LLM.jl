@use "./providers/abstract_provider" LLM Message SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Image Tool ToolCall ReasoningEffort ResponseFormat FinishReason Document json_schema
@use "./providers" OpenAI Anthropic Google Ollama
@use "./stream" from_json
@use "./models" get_pricing search_models search_providers

# Provider prefix → (config_key, env_var, constructor)
const PROVIDER_MAP = Dict(
  "anthropic" => (key="anthropic_key", env="ANTHROPIC_API_KEY", url=nothing),
  "openai"    => (key="openai_key",    env="OPENAI_API_KEY",    url=nothing),
  "google"    => (key="google_key",    env="GOOGLE_API_KEY",    url=nothing),
  "mistral"   => (key="mistral_key",   env="MISTRAL_API_KEY",   url="https://api.mistral.ai"),
  "deepseek"  => (key="deepseek_key",  env="DEEPSEEK_API_KEY",  url="https://api.deepseek.com"),
  "x-ai"     => (key="xai_key",       env="XAI_API_KEY",       url="https://api.x.ai"),
)

"Create an LLM instance from a model name string and config dict"
function LLM(model::String, config::Dict=Dict())
  lm = lowercase(model)
  # Handle provider-prefixed IDs like "x-ai/grok-4-fast" or "anthropic/claude-sonnet-4-6"
  if contains(lm, '/')
    prefix, name = split(lm, '/'; limit=2)
    if prefix == "ollama"
      return Ollama(String(name), get(config, "ollama_url", "http://localhost:11434"))
    end
    info = get(PROVIDER_MAP, prefix, nothing)
    if info !== nothing
      api_key = string(get(config, info.key, get(ENV, info.env, "")))
      if prefix == "anthropic"
        return Anthropic(model, api_key)
      elseif prefix == "google"
        return Google(model, api_key)
      elseif info.url !== nothing
        return OpenAI(model, api_key, info.url)
      else
        return OpenAI(model, api_key)
      end
    end
  end
  # Fallback: detect by model name prefix
  if startswith(lm, "ollama:")
    Ollama(model[length("ollama:")+1:end], get(config, "ollama_url", "http://localhost:11434"))
  elseif startswith(lm, "claude") || startswith(lm, "anthropic")
    Anthropic(model, string(get(config, "anthropic_key", get(ENV, "ANTHROPIC_API_KEY", ""))))
  elseif startswith(lm, "gpt") || startswith(lm, "o1") || startswith(lm, "o3") || startswith(lm, "o4")
    OpenAI(model, string(get(config, "openai_key", get(ENV, "OPENAI_API_KEY", ""))))
  elseif startswith(lm, "gemini")
    Google(model, string(get(config, "google_key", get(ENV, "GOOGLE_API_KEY", ""))))
  elseif startswith(lm, "mistral")
    OpenAI(model, string(get(config, "mistral_key", get(ENV, "MISTRAL_API_KEY", ""))), "https://api.mistral.ai")
  elseif startswith(lm, "deepseek")
    OpenAI(model, string(get(config, "deepseek_key", get(ENV, "DEEPSEEK_API_KEY", ""))), "https://api.deepseek.com")
  elseif startswith(lm, "grok")
    OpenAI(model, string(get(config, "xai_key", get(ENV, "XAI_API_KEY", ""))), "https://api.x.ai")
  else
    Ollama(model, get(config, "ollama_url", "http://localhost:11434"))
  end
end
