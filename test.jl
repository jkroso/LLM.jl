using Test
@use "./main" LLM
@use "./providers" OpenAI Anthropic Google Ollama
@use "./abstract_provider" SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Tool ToolCall ReasoningEffort ResponseFormat Message
@use "./pricing" get_pricing Mtoken token
@use "github.com/jkroso/Units.jl/Money" USD

@testset "LLM" begin
  config = Dict{String,Any}()
  @test LLM("claude-3-haiku", config) isa Anthropic
  @test LLM("anthropic-model", config) isa Anthropic
  @test LLM("gpt-4", config) isa OpenAI
  @test LLM("o1-mini", config) isa OpenAI
  @test LLM("o3-mini", config) isa OpenAI
  @test LLM("o4-mini", config) isa OpenAI
  @test LLM("gemini-pro", config) isa Google
  @test LLM("mistral-large", config) isa OpenAI
  @test LLM("deepseek-chat", config) isa OpenAI
  @test LLM("grok-2", config) isa OpenAI
  @test LLM("ollama:llama3", config) isa Ollama
  @test LLM("unknown-model", config) isa Ollama # defaults to Ollama
end

@testset "LLM base URLs" begin
  config = Dict{String,Any}()
  @test LLM("mistral-large", config).session.uri.host == "api.mistral.ai"
  @test LLM("deepseek-chat", config).session.uri.host == "api.deepseek.com"
  @test LLM("grok-2", config).session.uri.host == "api.x.ai"
  @test LLM("gpt-4", config).session.uri.host == "api.openai.com"
end

@testset "grok-4-1-fast-reasoning" begin
  llm = LLM("grok-4-1-fast-reasoning")
  stream = llm("You are a helpful assistant", "Reply with just the word 'hello'"; temperature=0.0)
  result = read(stream, String)
  @test occursin("hello", lowercase(result))
  @test stream.tokens[1] > token(0) # input tokens tracked
  @test stream.tokens[2] > token(0) # output tokens tracked
  close(llm)
end

@testset "get_pricing" begin
  @test get_pricing("nonexistent-model") == (0.0USD/Mtoken(1), 0.0USD/Mtoken(1))
  (input_price, output_price) = get_pricing("claude-sonnet-4-5-20250929")
  @test token(1_000_000) * input_price > 0.0USD
  @test token(1_000_000) * output_price > 0.0USD
end
