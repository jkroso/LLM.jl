@use "./providers/abstract_provider" LLM Message SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Image Tool ToolCall ReasoningEffort ResponseFormat FinishReason Document json_schema
@use "./providers" OpenAI Anthropic Google Ollama
@use "./stream" from_json
@use "./models" search

const PROVIDER_URLS = Dict(
  "mistral"  => "https://api.mistral.ai",
  "deepseek" => "https://api.deepseek.com",
  "xai"      => "https://api.x.ai",
)

"Look up the API key for a model from config dict or environment variables"
function get_api_key(info::NamedTuple, config::Dict)
  key = get(config, "$(info.provider)_key", nothing)
  key !== nothing && return string(key)
  for env in info.env
    val = get(ENV, env, nothing)
    val !== nothing && return val
  end
  ""
end

"Create an LLM instance from a model name string and config dict"
function LLM(model::String, config::Dict=Dict())
  provider = ""
  if contains(model, '/')
    provider, model = split(model, '/'; limit=2)
  end
  ap = isempty(provider) ? String[] : [provider]
  results = search("", model, allowed_providers=ap, max_results=1)
  @assert length(results) == 1 "Model '$model' not found"
  info = results[1]
  pid = info.provider
  if pid == "ollama"
    Ollama(info, get(config, "ollama_url", "http://localhost:11434"))
  elseif pid == "anthropic"
    Anthropic(info, get_api_key(info, config))
  elseif pid == "google"
    Google(info, get_api_key(info, config))
  else
    api_key = get_api_key(info, config)
    url = get(PROVIDER_URLS, pid, nothing)
    url !== nothing ? OpenAI(info, api_key, url) : OpenAI(info, api_key)
  end
end
