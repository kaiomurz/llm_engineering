from pathlib import Path

content = r'''# OpenAI SDK & API Cheat Sheet (Python)

> Practical reference for building applications with the OpenAI API. The Python SDK wraps the HTTP API; the **Responses API** is the recommended starting point for new text and multimodal applications. Model names, limits, and feature support change—check the official docs before production use.

## 1. Install and configure

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade openai
```

Set the key outside source control:

```bash
export OPENAI_API_KEY="your-api-key"   # macOS/Linux
# PowerShell:
$env:OPENAI_API_KEY="your-api-key"
```

Python (SDK reads `OPENAI_API_KEY` automatically):

```python
from openai import OpenAI

client = OpenAI()
```

For local development, you can load a `.env` file with `python-dotenv`; add `.env` to `.gitignore`. Never expose API keys in browser/mobile code, repositories, logs, or client-side bundles. Put calls behind your server.

## 2. The basic request

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="YOUR_MODEL_ID",
    instructions="Answer clearly and state uncertainty.",
    input="Explain a Python generator in two sentences.",
)

print(response.output_text)
```

Core parameters you will commonly use:

| Parameter | Purpose |
|---|---|
| `model` | Model ID available to your project |
| `input` | Text or structured multimodal/conversation input |
| `instructions` | High-level behaviour and constraints |
| `max_output_tokens` | Upper bound on generated output tokens |
| `tools` | Built-in tools or function tools the model may call |
| `tool_choice` | Let the model choose, require a tool, or restrict choice |
| `temperature` / `top_p` | Sampling controls where supported; usually change one, not both |
| `stream=True` | Receive incremental events |
| `previous_response_id` | Continue from a previous response |
| `store` | Controls response storage where supported; check data-retention requirements |

Do not assume every model supports every parameter. A rejected parameter is often a model/API compatibility issue.

## 3. Conversation turns and context

### Simple multi-turn conversation

Pass prior messages as input, or use response chaining. For explicit control over the history:

```python
history = [
    {"role": "user", "content": "What is a Python decorator?"},
]
r1 = client.responses.create(model="YOUR_MODEL_ID", input=history)
print(r1.output_text)

history += [
    {"role": "assistant", "content": r1.output_text},
    {"role": "user", "content": "Show a short example."},
]
r2 = client.responses.create(model="YOUR_MODEL_ID", input=history)
print(r2.output_text)
```

### Chain responses

```python
r1 = client.responses.create(
    model="YOUR_MODEL_ID",
    input="Give me three project ideas.",
)
r2 = client.responses.create(
    model="YOUR_MODEL_ID",
    previous_response_id=r1.id,
    input="Expand the second idea into a plan.",
)
print(r2.output_text)
```

Treat response chaining and storage as product/API behaviour, not a substitute for deciding what data you want retained. For stateless calls, send the relevant context yourself. For long conversations, trim or summarise history and keep only what is useful.

## 4. Streaming

Use streaming when the UI should show output as it arrives:

```python
stream = client.responses.create(
    model="YOUR_MODEL_ID",
    input="Explain vector search.",
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
print()
```

A stream contains multiple event types—not just text. Handle completion, failures, and tool events deliberately in production.

## 5. Async calls

Useful for async web servers and concurrent I/O:

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def main():
    response = await client.responses.create(
        model="YOUR_MODEL_ID",
        input="Return a one-line summary of RAG.",
    )
    print(response.output_text)

asyncio.run(main())
```

Use `asyncio.gather()` for independent requests, with a concurrency limit and sensible retry policy. Do not launch unbounded requests.

## 6. Structured output (reliable JSON shape)

Prefer Structured Outputs with a JSON Schema when the application needs a predictable shape; plain prompting for “valid JSON” is less reliable.

```python
from pydantic import BaseModel
from openai import OpenAI

class Classification(BaseModel):
    category: str
    confidence: float
    explanation: str

client = OpenAI()
response = client.responses.parse(
    model="YOUR_MODEL_ID",
    input=[
        {"role": "system", "content": "Classify the user's message."},
        {"role": "user", "content": "My package arrived broken."},
    ],
    text_format=Classification,
)

result = response.output_parsed
if result is not None:
    print(result.category, result.confidence)
```

Validate business rules after parsing too. A schema constrains format, not truth or correctness. Handle refusals and cases where parsed output is absent.

## 7. Tool / function calling

Function calling lets the model request that **your application** run a function. The model does not execute your Python function by itself.

```python
import json
from openai import OpenAI

client = OpenAI()

def get_weather(city: str) -> dict:
    # Replace with a real, trusted data source.
    return {"city": city, "temperature_c": 18}

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    "strict": True,
}]

response = client.responses.create(
    model="YOUR_MODEL_ID",
    input="What's the weather in London?",
    tools=tools,
)

# Execute tool calls, then send their outputs back to the model.
tool_outputs = []
for item in response.output:
    if item.type == "function_call" and item.name == "get_weather":
        args = json.loads(item.arguments)
        result = get_weather(**args)
        tool_outputs.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(result),
        })

if tool_outputs:
    final = client.responses.create(
        model="YOUR_MODEL_ID",
        previous_response_id=response.id,
        input=tool_outputs,
        tools=tools,
    )
    print(final.output_text)
else:
    print(response.output_text)
```

**Tool-calling checklist**
- Validate arguments and permissions in your code; never trust model output as authorization.
- Use allowlists for operations and resources.
- Require explicit confirmation for consequential actions (payments, deletion, external messages).
- Set timeouts, limit tool calls, and log tool name + outcome (avoid logging secrets).
- Return concise, structured tool results.
- Follow the documented tool-call loop until the model produces a final answer or you reach a safe limit.

## 8. Built-in tools

The Responses API can support built-in tools such as web search, file search, and code interpreter, subject to model/account availability. Example using web search:

```python
response = client.responses.create(
    model="YOUR_MODEL_ID",
    input="Find recent official guidance on Python's supported versions.",
    tools=[{"type": "web_search"}],
)
print(response.output_text)
```

For private documents, use file search / vector stores where appropriate. For computation on uploaded data, check Code Interpreter support. Tool availability, syntax, and billing can vary—use the linked guide for the exact current configuration.

## 9. Images and files

### Image input by URL

```python
response = client.responses.create(
    model="YOUR_MODEL_ID",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Describe this image."},
            {"type": "input_image",
             "image_url": "https://example.com/image.jpg"},
        ],
    }],
)
print(response.output_text)
```

Use a reachable image URL or the documented base64/data-URL approach. Do not assume the model can access a local file path.

### Upload a file

```python
uploaded = client.files.create(
    file=open("document.pdf", "rb"),
    purpose="user_data",
)
print(uploaded.id)
```

Upload purpose and accepted file types depend on the feature. Follow the relevant guide for attaching a file to a Responses request or creating a vector store; uploading a file alone does not automatically make it searchable by the model.

## 10. Embeddings and semantic search

Embeddings map text to vectors for similarity search, clustering, recommendations, and retrieval-augmented generation (RAG).

```python
response = client.embeddings.create(
    model="YOUR_EMBEDDING_MODEL_ID",
    input=[
        "The cat sat on the mat.",
        "A kitten rested on a rug.",
    ],
)
vectors = [item.embedding for item in response.data]
```

A minimal RAG flow:
1. Split source documents into useful chunks and retain metadata/source IDs.
2. Embed chunks and store vectors in a vector index/database.
3. Embed the user query and retrieve top relevant chunks.
4. Send the retrieved excerpts to the model with instructions to answer from those sources.
5. Cite source IDs and say when the retrieved evidence is insufficient.

Use the same embedding model for indexing and queries. Choose chunk size, overlap, top-k, and reranking based on evaluation—not guesswork. Treat retrieved text as untrusted data, not instructions.

## 11. Image generation, audio, and real-time

These are separate feature families with their own endpoints and payload formats:
- **Image generation/editing:** see the Images and Image Generation guides.
- **Speech generation:** audio speech endpoint.
- **Transcription:** audio transcription endpoint.
- **Realtime voice/audio:** Realtime API; often WebSocket/WebRTC-based.

Do not assume the text-generation request shape works for these. Consult the relevant guide and check supported models, formats, and limits.

## 12. Legacy Chat Completions API

Chat Completions remains supported and is common in existing code. For new general-purpose projects, start with Responses unless a specific compatibility need points elsewhere.

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="YOUR_MODEL_ID",
    messages=[
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is a primary key?"},
    ],
)
print(completion.choices[0].message.content)
```

Do not mix the two API response shapes: Responses commonly uses `response.output_text`; Chat Completions uses `completion.choices[0].message.content`.

## 13. Errors, retries, timeouts, and observability

```python
import openai
from openai import OpenAI

client = OpenAI(timeout=30.0, max_retries=2)

try:
    response = client.responses.create(
        model="YOUR_MODEL_ID",
        input="Hello",
    )
except openai.AuthenticationError:
    print("Check API key / project credentials.")
except openai.RateLimitError:
    print("Rate limited or quota/billing issue; inspect the error details.")
except openai.APIConnectionError:
    print("Network or connection problem.")
except openai.APIStatusError as exc:
    print("HTTP status:", exc.status_code)
    print("Request ID:", exc.request_id)
```

Useful habits:
- SDK retries selected transient failures by default; configure `max_retries` and `timeout` intentionally.
- Retry transient 429/5xx/network failures with backoff; do not blindly retry invalid requests or authentication failures.
- Record request IDs, latency, model ID, token usage, and outcome.
- Monitor rate limits, token usage, spend, and error rates.
- Handle partial/failed streams and tool errors separately.
- Do not assume a timeout means the server did no work; design idempotent operations where needed.

## 14. Cost and latency controls

- Use the smallest capable model for routine tasks; reserve more capable models for tasks that need them.
- Limit input context and `max_output_tokens`.
- Cache stable prompts/results where appropriate; reuse repeated prompt prefixes when supported.
- Batch offline workloads when the Batch API fits the latency requirements.
- Stream for perceived latency, not necessarily lower total compute cost.
- Track input, output, cached, and tool-related usage as applicable; consult current pricing.
- Evaluate quality against a representative test set before changing model or prompt.

Inspect usage from the returned response when available:

```python
response = client.responses.create(
    model="YOUR_MODEL_ID",
    input="Summarise this in one sentence.",
)
print(response.usage)
```

## 15. Production checklist

- [ ] API key is stored in a secret manager/environment, never in client code.
- [ ] Model ID is configurable, not scattered as a literal through the codebase.
- [ ] Inputs are validated and bounded; outputs are treated as untrusted.
- [ ] Tool calls have authorization, validation, timeouts, and audit logs.
- [ ] Refusals, empty output, malformed/unsupported cases, and API errors are handled.
- [ ] Rate limits, concurrency, retries, and timeouts are configured.
- [ ] Sensitive data handling, retention, and access controls are reviewed.
- [ ] Prompts and model changes are tested with a repeatable evaluation set.
- [ ] Costs, latency, usage, and failures are monitored.
- [ ] Dependency versions are pinned and upgraded deliberately.

## 16. Official references

- [OpenAI API documentation](https://platform.openai.com/docs)
- [API reference](https://platform.openai.com/docs/api-reference)
- [Python SDK repository and README](https://github.com/openai/openai-python)
- [Responses API guide](https://platform.openai.com/docs/guides/responses)
- [Text generation](https://platform.openai.com/docs/guides/text)
- [Function calling](https://platform.openai.com/docs/guides/function-calling)
- [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Tools overview](https://platform.openai.com/docs/guides/tools)
- [Images and vision](https://platform.openai.com/docs/guides/images-vision)
- [Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Audio](https://platform.openai.com/docs/guides/audio)
- [Rate limits](https://platform.openai.com/docs/guides/rate-limits)
- [Pricing](https://openai.com/api/pricing/)

**Rule of thumb:** start with `client.responses.create(...)`; add structured outputs when you need a schema, tools when the model must interact with your code or data, and retrieval when it needs grounded access to a document collection.
'''
path = Path("/mnt/data/openai_sdk_api_cheat_sheet.md")
path.write_text(content, encoding="utf-8")
print(f"Created {path} ({path.stat().st_size:,} bytes)")
