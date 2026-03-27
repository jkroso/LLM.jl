using Test
@use "./main" LLM
@use "./providers" OpenAI Anthropic Google Ollama
@use "./abstract_provider" SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Image Tool ToolCall ReasoningEffort ResponseFormat Message
@use "./pricing" get_pricing Mtoken token
@use "github.com/jkroso/Units.jl/Money" USD
@use "./providers/openai" to_openai

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

@testset "to_openai serialization" begin
  # SystemMessage
  @test to_openai(SystemMessage("be helpful")) == Dict("role" => "system", "content" => "be helpful")

  # UserMessage — text only
  @test to_openai(UserMessage("hello")) == Dict("role" => "user", "content" => "hello")

  # UserMessage — with image URL
  msg = UserMessage("what's this?", [ImageURL("https://example.com/img.png")])
  result = to_openai(msg)
  @test result["role"] == "user"
  @test result["content"] isa Vector
  @test length(result["content"]) == 2
  @test result["content"][1] == Dict("type" => "text", "text" => "what's this?")
  @test result["content"][2] == Dict("type" => "image_url", "image_url" => Dict("url" => "https://example.com/img.png", "detail" => "auto"))

  # UserMessage — with image data
  img_data = ImageData(UInt8[0xff, 0xd8], "image/jpeg", "high")
  msg = UserMessage("describe", [img_data])
  result = to_openai(msg)
  @test result["content"][2]["type"] == "image_url"
  @test startswith(result["content"][2]["image_url"]["url"], "data:image/jpeg;base64,")
  @test result["content"][2]["image_url"]["detail"] == "high"

  # UserMessage — with audio
  aud = Audio(UInt8[0x00, 0x01], "mp3")
  msg = UserMessage("transcribe", Image[], [aud])
  result = to_openai(msg)
  @test result["content"][2]["type"] == "input_audio"
  @test result["content"][2]["input_audio"]["format"] == "mp3"

  # AIMessage — text only
  @test to_openai(AIMessage("hello")) == Dict{String,Any}("role" => "assistant", "content" => "hello")

  # AIMessage — with tool calls
  tc = ToolCall("call_123", "get_weather", Dict("city" => "Boston"))
  msg = AIMessage("", [tc])
  result = to_openai(msg)
  @test result["role"] == "assistant"
  @test result["content"] === nothing
  @test length(result["tool_calls"]) == 1
  @test result["tool_calls"][1]["id"] == "call_123"
  @test result["tool_calls"][1]["type"] == "function"
  @test result["tool_calls"][1]["function"]["name"] == "get_weather"

  # ToolResultMessage
  @test to_openai(ToolResultMessage("call_123", "72°F")) == Dict("role" => "tool", "tool_call_id" => "call_123", "content" => "72°F")

  # Tool
  params = Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")), "required" => ["city"])
  tool = Tool("get_weather", "Get the weather", params)
  result = to_openai(tool)
  @test result == Dict("type" => "function", "function" => Dict("name" => "get_weather", "description" => "Get the weather", "parameters" => params))
end
