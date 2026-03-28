@use "./providers/abstract_provider" LLM Message SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Image Tool ToolCall ReasoningEffort ResponseFormat FinishReason Document json_schema
@use "./providers" OpenAI Anthropic Google Ollama
@use "./stream" from_json
@use "./models" get_pricing search_models search_providers

"Create an LLM instance from a model name string and config dict"
function LLM(model::String, config::Dict=Dict())
  lm = lowercase(model)
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
