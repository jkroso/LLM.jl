@use "./providers" OpenAI Anthropic Google Ollama XAI
@use "./providers/abstract_provider" SystemMessage UserMessage AIMessage ToolResultMessage ImageURL ImageData Audio Image Tool ToolCall ReasoningEffort ResponseFormat Message FinishReason Document json_schema
@use "./models" search Mtoken token
@use "github.com/jkroso/Units.jl/Money" USD
@use "github.com/jkroso/JSON.jl/write" JSON
@use "./providers/anthropic" to_anthropic
@use "./providers/openai" to_openai
@use "./providers/ollama" to_ollama
@use "./providers/xai" to_xai make_xai_parser
@use "./stream" TokenStream sse from_json
@use "./models" token
@use "github.com/jkroso/HTTP.jl/client" Response Header
@use "." LLM
@use Test...

@use Base64...

struct TestPerson
  name::String
  age::Int
end

struct TestGreeting
  greeting::String
end

@testset "LLM" begin
  @test LLM("anthropic/claude-haiku-4-5") isa Anthropic
  @test LLM("openai/gpt-4o") isa OpenAI
  @test LLM("openai/o3-mini") isa OpenAI
  @test LLM("google/gemini-2.0-flash") isa Google
  @test LLM("mistral/mistral-large-latest") isa OpenAI
  @test LLM("deepseek/deepseek-chat") isa OpenAI
  @test LLM("xai/grok-2") isa XAI
  @test LLM("ollama/gemma4:31b") isa Ollama
  @test LLM("anthropic/claude-haiku-4-5").info.provider == "anthropic"
  @test LLM("xai/grok-2").info.provider == "xai"
  @test LLM("mistral/mistral-large-latest").info.provider == "mistral"
end

@testset "LLM base URLs" begin
  @test LLM("mistral/mistral-large-latest").session.uri.host == "api.mistral.ai"
  @test LLM("deepseek/deepseek-chat").session.uri.host == "api.deepseek.com"
  @test LLM("xai/grok-2").session.uri.host == "api.x.ai"
  @test LLM("openai/gpt-4o").session.uri.host == "api.openai.com"
end


@testset "search" begin
  results = search("", "claude", allowed_providers="anthropic")
  @test length(results) > 0
  @test all(r -> occursin("claude", lowercase(r.id)) || occursin("claude", lowercase(r.name)), results)
  @test all(r -> hasproperty(r, :provider) && hasproperty(r, :id) && hasproperty(r, :pricing), results)

  # sorted newest first
  dates = [r.release_date for r in results if !isempty(r.release_date)]
  @test issorted(dates, rev=true)

  # filter by provider only
  results = search("openai", "", allowed_providers="openai")
  @test length(results) > 0
  @test all(r -> r.provider == "openai", results)

  # single-arg OR search matches provider or model
  results = search("openai", allowed_providers="openai")
  @test length(results) > 0
  @test all(r -> occursin("openai", lowercase(r.provider)) || occursin("openai", lowercase(r.id)), results)

  # combine provider and model queries
  results = search("openai", "gpt", allowed_providers="openai")
  @test length(results) > 0
  @test all(r -> r.provider == "openai" && occursin("gpt", lowercase(r.id)), results)

  # provider + model narrows to exact intersection
  results = search("gemma", allowed_providers="ollama")
  @test length(results) == 1
  @test results[1].provider == "ollama"
  @test occursin("gemma", lowercase(results[1].id))

  # allowed_providers — exact match, single string
  results = search(allowed_providers="openai")
  @test length(results) > 0
  @test all(r -> r.provider == "openai", results)

  # allowed_providers — exact match, multiple providers
  results = search(allowed_providers=["anthropic", "openai"])
  @test length(results) > 0
  @test all(r -> r.provider in ("anthropic", "openai"), results)

  # allowed_providers — combined with model query
  results = search("gpt", allowed_providers="openai")
  @test length(results) > 0
  @test all(r -> r.provider == "openai" && occursin("gpt", lowercase(r.id)), results)

  # filter by reasoning
  results = search("", "", allowed_providers=["anthropic", "openai"], reasoning=true, max_results=5)
  @test all(r -> r.reasoning == true, results)

  # filter by vision
  results = search("", "", allowed_providers=["anthropic", "openai"], vision=true, max_results=5)
  @test all(r -> "image" in r.modalities.input, results)

  # no results for nonsense query
  @test isempty(search("", "zzz_nonexistent_model_xyz", allowed_providers="openai"))

  # local ollama models show up in search results
  ollama_results = search("", "", allowed_providers="ollama", max_results=100)
  @test length(ollama_results) > 0
  @test all(r -> r.provider == "ollama", ollama_results)
  @test all(r -> hasproperty(r, :id) && hasproperty(r, :name) && hasproperty(r, :modalities), ollama_results)
  # searching by name should also find local ollama models
  first_model = ollama_results[1].id
  name_results = search("", first_model, allowed_providers="ollama", max_results=100)
  @test any(r -> r.provider == "ollama" && r.id == first_model, name_results)
end

@testset "pricing" begin
  results = search("", "claude-haiku-4-5", allowed_providers="anthropic", max_results=1)
  (input_price, output_price) = results[1].pricing
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

@testset "to_ollama serialization" begin
  # SystemMessage
  @test to_ollama(SystemMessage("be helpful")) == Dict("role" => "system", "content" => "be helpful")

  # UserMessage — text only
  @test to_ollama(UserMessage("hello")) == Dict{String,Any}("role" => "user", "content" => "hello")

  # UserMessage — with image data (base64 in images array)
  img_data = ImageData(UInt8[0xff, 0xd8], "image/jpeg")
  msg = UserMessage("describe", [img_data])
  result = to_ollama(msg)
  @test result["role"] == "user"
  @test result["content"] == "describe"
  @test length(result["images"]) == 1
  @test result["images"][1] == base64encode(UInt8[0xff, 0xd8])

  # UserMessage — with image URL (should error)
  msg = UserMessage("what's this?", [ImageURL("https://example.com/img.png")])
  @test_throws ErrorException to_ollama(msg)

  # UserMessage — with audio (should error)
  aud = Audio(UInt8[0x00], "mp3")
  msg = UserMessage("transcribe", Image[], [aud])
  @test_throws ErrorException to_ollama(msg)

  # UserMessage — with documents (should error)
  doc = Document(UInt8[0x25], "application/pdf")
  msg = UserMessage("read", Image[], Audio[], [doc])
  @test_throws ErrorException to_ollama(msg)

  # AIMessage — text only
  @test to_ollama(AIMessage("hello")) == Dict{String,Any}("role" => "assistant", "content" => "hello")

  # AIMessage — with tool calls
  tc = ToolCall("call_123", "get_weather", Dict("city" => "Boston"))
  msg = AIMessage("", [tc])
  result = to_ollama(msg)
  @test result["role"] == "assistant"
  @test result["content"] == ""
  @test length(result["tool_calls"]) == 1
  @test result["tool_calls"][1]["function"]["name"] == "get_weather"
  @test result["tool_calls"][1]["function"]["arguments"] == Dict("city" => "Boston")

  # ToolResultMessage
  @test to_ollama(ToolResultMessage("call_123", "72°F")) == Dict("role" => "tool", "content" => "72°F")

  # Tool
  params = Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")), "required" => ["city"])
  tool = Tool("get_weather", "Get the weather", params)
  result = to_ollama(tool)
  @test result == Dict("type" => "function", "function" => Dict("name" => "get_weather", "description" => "Get the weather", "parameters" => params))
end

@testset "to_anthropic serialization" begin
  # UserMessage — text only
  @test to_anthropic(UserMessage("hello")) == Dict("role" => "user", "content" => "hello")

  # UserMessage — with image URL
  msg = UserMessage("what's this?", [ImageURL("https://example.com/img.png")])
  result = to_anthropic(msg)
  @test result["role"] == "user"
  @test result["content"] isa Vector
  @test result["content"][1] == Dict("type" => "text", "text" => "what's this?")
  @test result["content"][2] == Dict("type" => "image", "source" => Dict("type" => "url", "url" => "https://example.com/img.png"))

  # UserMessage — with image data
  img_data = ImageData(UInt8[0xff, 0xd8], "image/jpeg", "high")
  msg = UserMessage("describe", [img_data])
  result = to_anthropic(msg)
  @test result["content"][2]["type"] == "image"
  @test result["content"][2]["source"]["type"] == "base64"
  @test result["content"][2]["source"]["media_type"] == "image/jpeg"

  # UserMessage — with document
  doc = Document(UInt8[0x25, 0x50], "application/pdf")
  msg = UserMessage("read this", Image[], Audio[], [doc])
  result = to_anthropic(msg)
  @test result["content"][2]["type"] == "document"
  @test result["content"][2]["source"]["type"] == "base64"
  @test result["content"][2]["source"]["media_type"] == "application/pdf"

  # UserMessage — with audio (should error)
  aud = Audio(UInt8[0x00], "mp3")
  msg = UserMessage("transcribe", Image[], [aud], Document[])
  @test_throws ErrorException to_anthropic(msg)

  # AIMessage — text only
  @test to_anthropic(AIMessage("hello")) == Dict("role" => "assistant", "content" => "hello")

  # AIMessage — with tool calls
  tc = ToolCall("toolu_123", "get_weather", Dict("city" => "Boston"))
  msg = AIMessage("Let me check.", [tc])
  result = to_anthropic(msg)
  @test result["role"] == "assistant"
  @test result["content"] isa Vector
  @test result["content"][1] == Dict("type" => "text", "text" => "Let me check.")
  @test result["content"][2]["type"] == "tool_use"
  @test result["content"][2]["id"] == "toolu_123"
  @test result["content"][2]["name"] == "get_weather"
  @test result["content"][2]["input"] == Dict("city" => "Boston")

  # ToolResultMessage
  result = to_anthropic(ToolResultMessage("toolu_123", "72°F"))
  @test result["role"] == "user"
  @test result["content"][1]["type"] == "tool_result"
  @test result["content"][1]["tool_use_id"] == "toolu_123"
  @test result["content"][1]["content"] == "72°F"

  # Tool
  params = Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")), "required" => ["city"])
  tool = Tool("get_weather", "Get the weather", params)
  result = to_anthropic(tool)
  @test result == Dict("name" => "get_weather", "description" => "Get the weather", "input_schema" => params)
end

@testset "json_schema" begin
  @test json_schema(String) == Dict("type" => "string")
  @test json_schema(Bool) == Dict("type" => "boolean")
  @test json_schema(Int) == Dict("type" => "integer")
  @test json_schema(Float64) == Dict("type" => "number")
  @test json_schema(Vector{String}) == Dict("type" => "array", "items" => Dict("type" => "string"))

  schema = json_schema(TestPerson)
  @test schema["type"] == "object"
  @test schema["properties"]["name"] == Dict("type" => "string")
  @test schema["properties"]["age"] == Dict("type" => "integer")
  @test Set(schema["required"]) == Set(["name", "age"])
  @test schema["additionalProperties"] == false
end

@testset "to_xai serialization" begin
  # UserMessage — text only
  @test to_xai(UserMessage("hello")) == Dict("type" => "message", "role" => "user", "content" => "hello")

  # UserMessage — with image URL
  msg = UserMessage("what's this?", [ImageURL("https://example.com/img.png")])
  result = to_xai(msg)
  @test result["type"] == "message"
  @test result["content"][1] == Dict("type" => "input_text", "text" => "what's this?")
  @test result["content"][2] == Dict("type" => "input_image", "image_url" => "https://example.com/img.png")

  # AIMessage
  @test to_xai(AIMessage("hi")) == Dict("type" => "message", "role" => "assistant", "content" => "hi")

  # ToolResultMessage
  @test to_xai(ToolResultMessage("call_1", "42")) == Dict("type" => "function_call_output", "call_id" => "call_1", "output" => "42")

  # Tool
  params = Dict("type" => "object", "properties" => Dict("x" => Dict("type" => "number")))
  tool = Tool("calc", "Calculate", params)
  @test to_xai(tool) == Dict("type" => "function", "name" => "calc", "description" => "Calculate", "parameters" => params)
end

@testset "xAI parser" begin
  llm = XAI((id="grok-4", provider="xai", env=String[], name="Grok 4", pricing=(0,0), release_date="", reasoning=false, modalities=(input=String[], output=String[])), "fake")

  # Helper to create a TokenStream without a real HTTP response
  function make_test_stream(llm)
    parser = sse(make_xai_parser(llm))
    resp = Response(Int16(200), Header(), IOBuffer())
    TokenStream(resp, parser)
  end

  # Build a stream and feed it SSE lines simulating a code_interpreter response
  s = make_test_stream(llm)

  # response.created — captures response ID
  s.parse_line(s, "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_abc123\"}}")
  @test llm.last_response_id == "resp_abc123"

  # text deltas
  s.parse_line(s, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"The answer is \"}")
  s.parse_line(s, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"22,791,481\"}")
  @test String(take!(s.buf)) == "The answer is 22,791,481"

  # function_call tool result
  s.parse_line(s, """data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_99","name":"code_interpreter","arguments":"{\\\"code\\\":\\\"3847 * 5923\\\"}"}}""")
  @test length(s.tool_calls) == 1
  @test s.tool_calls[1].name == "code_interpreter"
  @test s.tool_calls[1].arguments["code"] == "3847 * 5923"

  # response.completed — sets usage and done
  s.parse_line(s, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_abc123\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":25}}}")
  @test s.done == true
  @test s.finish_reason == FinishReason.stop
  @test s.tokens[1] == token(10)
  @test s.tokens[2] == token(25)

  # incomplete response sets length finish reason
  s2 = make_test_stream(llm)
  s2.parse_line(s2, """data: {"type":"response.completed","response":{"id":"resp_xyz","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":5,"output_tokens":100}}}""")
  @test s2.finish_reason == FinishReason.length
  @test s2.done == true
end
