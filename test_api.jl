@use "./providers/abstract_provider" SystemMessage UserMessage AIMessage ToolResultMessage Tool ToolCall ReasoningEffort ResponseFormat Message FinishReason
@use "github.com/jkroso/JSON.jl/write" JSON
@use "./stream" from_json
@use "./models" token
@use "." LLM
@use Test...

struct TestPerson
  name::String
  age::Int
end

struct TestGreeting
  greeting::String
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

@testset "multi-turn conversation" begin
  llm = LLM("grok-4-1-fast-reasoning")
  messages = Message[
    SystemMessage("You are a helpful assistant"),
    UserMessage("My name is Alice"),
    AIMessage("Nice to meet you, Alice!"),
    UserMessage("What is my name?")
  ]
  stream = llm(messages; temperature=0.0)
  result = read(stream, String)
  @test occursin("Alice", result)
  @test stream.finish_reason == FinishReason.stop
  close(llm)
end

@testset "tool calling" begin
  llm = LLM("grok-4-1-fast-reasoning")
  tools = [Tool("get_temperature", "Get the current temperature in a city", Dict(
    "type" => "object",
    "properties" => Dict("city" => Dict("type" => "string", "description" => "City name")),
    "required" => ["city"]))]
  messages = Message[
    SystemMessage("Use the get_temperature tool to answer weather questions"),
    UserMessage("What's the temperature in Boston?")
  ]
  stream = llm(messages; temperature=0.0, tools=tools)
  read(stream, String) # drain the stream
  @test stream.finish_reason == FinishReason.tool_calls
  @test length(stream.tool_calls) >= 1
  @test stream.tool_calls[1].name == "get_temperature"
  @test haskey(stream.tool_calls[1].arguments, "city")
  close(llm)
end

@testset "structured output" begin
  llm = LLM("grok-4-1-fast-reasoning")
  messages = Message[
    SystemMessage("Return a JSON object with a 'greeting' field"),
    UserMessage("Say hello")
  ]
  stream = llm(messages; temperature=0.0, response_format=ResponseFormat.json)
  result = read(stream, JSON)
  @test result isa Dict
  @test haskey(result, "greeting")
  @test stream.finish_reason == FinishReason.stop
  close(llm)
end

@testset "ollama basic" begin
  llm = LLM("ollama/Qwen3:14b")
  stream = llm("You are a helpful assistant", "Reply with just the word 'hello'"; temperature=0.0)
  result = read(stream, String)
  @test occursin("hello", lowercase(result))
  @test stream.tokens[1] > token(0)
  @test stream.tokens[2] > token(0)
  close(llm)
end

@testset "ollama multi-turn conversation" begin
  llm = LLM("ollama/Qwen3:14b")
  messages = Message[
    SystemMessage("You are a helpful assistant. Be concise."),
    UserMessage("My name is Alice"),
    AIMessage("Nice to meet you, Alice!"),
    UserMessage("What is my name?")
  ]
  stream = llm(messages; temperature=0.0)
  result = read(stream, String)
  @test occursin("Alice", result)
  @test stream.finish_reason == FinishReason.stop
  close(llm)
end

@testset "ollama tool calling" begin
  llm = LLM("ollama/Qwen3:14b")
  tools = [Tool("get_temperature", "Get the current temperature in a city", Dict(
    "type" => "object",
    "properties" => Dict("city" => Dict("type" => "string", "description" => "City name")),
    "required" => ["city"]))]
  messages = Message[
    SystemMessage("Use the get_temperature tool to answer weather questions. Always use the tool."),
    UserMessage("What's the temperature in Boston?")
  ]
  stream = llm(messages; temperature=0.0, tools=tools)
  read(stream, String)
  @test stream.finish_reason == FinishReason.tool_calls
  @test length(stream.tool_calls) >= 1
  @test stream.tool_calls[1].name == "get_temperature"
  @test haskey(stream.tool_calls[1].arguments, "city")
  close(llm)
end

@testset "ollama structured output" begin
  llm = LLM("ollama/Qwen3:14b")
  messages = Message[
    SystemMessage("Return a JSON object with a 'greeting' field"),
    UserMessage("Say hello")
  ]
  stream = llm(messages; temperature=0.0, response_format=ResponseFormat.json)
  result = read(stream, JSON)
  @test result isa Dict
  @test haskey(result, "greeting")
  close(llm)
end

@testset "ollama return_type" begin
  llm = LLM("ollama/Qwen3:14b")
  result = from_json(llm("Make up a person"; return_type=TestPerson), TestPerson)
  @test result isa TestPerson
  @test !isempty(result.name)
  @test result.age isa Int
  close(llm)
end

@testset "ollama reasoning" begin
  llm = LLM("ollama/Qwen3:14b")
  stream = llm("What is 23 * 47?"; reasoning_effort=ReasoningEffort.high)
  result = read(stream, String)
  @test occursin("1081", result)
  @test !isempty(String(take!(stream.thinking)))
  @test stream.tokens[2] > token(0)
  close(llm)
end

@testset "anthropic multi-turn conversation" begin
  llm = LLM("claude-haiku-4-5")
  messages = Message[
    SystemMessage("You are a helpful assistant"),
    UserMessage("My name is Alice"),
    AIMessage("Nice to meet you, Alice!"),
    UserMessage("What is my name?")
  ]
  stream = llm(messages; temperature=0.0)
  result = read(stream, String)
  @test occursin("Alice", result)
  @test stream.finish_reason == FinishReason.stop
  close(llm)
end

@testset "anthropic tool calling" begin
  llm = LLM("claude-haiku-4-5")
  tools = [Tool("get_temperature", "Get the current temperature in a city", Dict(
    "type" => "object",
    "properties" => Dict("city" => Dict("type" => "string", "description" => "City name")),
    "required" => ["city"]))]
  messages = Message[
    SystemMessage("Use the get_temperature tool to answer weather questions. Always use the tool."),
    UserMessage("What's the temperature in Boston?")
  ]
  stream = llm(messages; temperature=0.0, tools=tools)
  read(stream, String) # drain the stream
  @test stream.finish_reason == FinishReason.tool_calls
  @test length(stream.tool_calls) >= 1
  @test stream.tool_calls[1].name == "get_temperature"
  @test haskey(stream.tool_calls[1].arguments, "city")
  close(llm)
end

@testset "anthropic structured output" begin
  llm = LLM("claude-haiku-4-5")
  messages = Message[
    SystemMessage("Return a greeting"),
    UserMessage("Say hello")
  ]
  stream = llm(messages; temperature=0.0, return_type=TestGreeting)
  result = from_json(stream, TestGreeting)
  @test result isa TestGreeting
  @test !isempty(result.greeting)
  @test stream.finish_reason == FinishReason.stop
  close(llm)
end

@testset "anthropic return_type with single-string call" begin
  llm = LLM("claude-haiku-4-5")
  result = from_json(llm("Make up a person"; return_type=TestPerson), TestPerson)
  @test result isa TestPerson
  @test !isempty(result.name)
  @test result.age isa Int
  close(llm)
end

@testset "anthropic unsupported features" begin
  llm = LLM("claude-haiku-4-5")
  messages = Message[SystemMessage("test"), UserMessage("test")]
  @test_throws ErrorException llm(messages; response_format=ResponseFormat.json)
  close(llm)
end
