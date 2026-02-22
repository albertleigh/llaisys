# llaisys-infer

OpenAI-compatible chat-completion inference service powered by the llaisys engine.

## Quick Start

```bash
# From the project root (assuming .venv is already active with llaisys installed)
cd services/infer
pip install -e .

# Start the server
uvicorn infer.main:app --host 0.0.0.0 --port 8000

# Or use the CLI entry point
llaisys-infer --model ../../models/DeepSeek-R1-Distill-Qwen-1.5B --device cpu
# Or
llaisys-infer --model ../../models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia
```

## API

### POST /v1/chat/completions

Follows the [OpenAI Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create) format.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distill-qwen-1.5b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "stream": false
  }'
```

### Streaming

Set `"stream": true` to receive Server-Sent Events (SSE):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distill-qwen-1.5b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### GET /v1/models

List available models.
