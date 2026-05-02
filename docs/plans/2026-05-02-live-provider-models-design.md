# Live Provider Model Information Design

## Goal

Make provider APIs the primary source for current model availability, while keeping `models.dev` as the enrichment source for pricing and other metadata that providers do not consistently expose.

## Approach

Use a hybrid live-primary source.

Provider APIs decide which model IDs exist. The local `models.dev` data enriches those records by `(provider, id)` with pricing, logos, env vars, release date, modality, context, reasoning, tool-call, and temperature metadata where available.

When a live provider model is missing from `models.dev`, it still appears in search results with conservative defaults:

- `pricing = zero_price`
- `env` from provider-level defaults
- `name = id`
- `release_date = ""`
- `reasoning = false`
- `tool_call = false`
- `temperature = true`
- `modalities = (input=["text"], output=["text"])`
- `vision = false`
- `context = nothing`

## Public API

Preserve the current search and constructor surface:

- `search(provider, model; ...)`
- `search(query; ...)`
- `LLM(model, config)`

The search result named tuple shape stays unchanged so existing provider constructors continue to work.

The default model source should become live-primary where a direct provider fetcher exists. Providers without a reliable direct fetcher can continue to use `models.dev` until support is added.

## Provider Fetchers

Add a small provider-fetcher layer that returns minimal live records with at least:

- provider ID
- model ID
- display name when available

Initial targets:

- OpenAI-compatible providers: `GET /v1/models`
- Ollama: keep using `/api/tags`
- Google: use the Gemini model listing endpoint
- Anthropic: add if their model listing endpoint is available and stable; otherwise leave on the existing `models.dev` fallback initially

OpenAI-compatible fetchers need provider-specific base URLs and authentication behavior. If a provider cannot be fetched because credentials are absent or the endpoint fails, search should fall back to enriched `models.dev` records for that provider rather than failing globally.

## Data Flow

1. Load the `models.dev` registry and build the enrichment index.
2. For each requested provider, ask the live fetcher for current model IDs.
3. Convert live records to the existing result shape.
4. Merge enrichment fields from the registry when a matching `(provider, id)` exists.
5. Sort results newest-first when release dates are known, keeping unknown dates after known dates.
6. Apply existing provider, model, reasoning, and vision filters.

`models.dev` remains cached and refreshed on the existing schedule. Live provider results can be cached briefly in memory during a process to avoid repeated network calls during a single search flow, but the cache should be easy to bypass or expire quickly.

## Error Handling

Live fetch failures should be provider-local. A failed OpenAI fetch should not prevent Anthropic or Ollama results from loading.

Fallback rules:

- Missing credentials: use `models.dev` for that provider.
- Network error or non-2xx response: use `models.dev` for that provider.
- Malformed provider response: use `models.dev` for that provider.
- Missing enrichment: keep the live model with defaults.

The implementation should avoid printing warnings during normal search. Tests can assert fallback behavior directly.

## Testing

Use test-first changes around the model-source layer.

Core tests:

- A live model that exists in `models.dev` is returned with `models.dev` pricing.
- A live model missing from `models.dev` is returned with defaults and zero pricing.
- A model present only in `models.dev` is not returned when the live provider fetch succeeds.
- If live fetch fails, provider results fall back to `models.dev`.
- Existing `search(...)` filters still work after live-primary normalization.
- `LLM(provider/model)` can construct a provider from a live-enriched result.

Network calls should be abstracted enough that tests can pass fake fetchers or fake response data rather than calling real provider APIs.
