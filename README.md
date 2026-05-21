# GitBlock — Free AI Inference Network

Free AI for everyone. Powered by the community.

## What is GitBlock?

GitBlock is a decentralized AI inference network where:
- **Users** get free AI inference (Llama, Mistral, Gemma, 80+ models)
- **Node operators** earn rewards for serving models
- **The community** governs the network

No paywalls. No credit cards. No gatekeepers. Just open intelligence.

## Quick Start

### Install

```bash
pip install gitblock
```

### Get an API Key

Connect your wallet at [gitblock.org](https://gitblock.org) to generate a free API key. No email, no signup.

### Use the SDK

```python
from gitblock import GitBlock

client = GitBlock(api_key="gb_free_xxx")

response = client.chat(
    model="llama-3.3-70b",
    messages=[{"role": "user", "content": "Explain recursion"}]
)
print(response.choices[0].message.content)
```

### Use the CLI

```bash
# One-shot question
gitblock ask "What is machine learning?"

# Interactive chat
gitblock chat --model llama-3.3-70b

# List available models
gitblock models

# Set your API key
gitblock auth --key gb_free_xxx
```

### Streaming

```python
stream = client.chat_stream(
    model="mistral-7b",
    messages=[{"role": "user", "content": "Write a haiku about code"}]
)
for chunk in stream:
    print(chunk.delta, end="", flush=True)
```

### REST API (any language)

```bash
curl -X POST https://api.gitblock.io/v1/chat/completions \
  -H "Authorization: Bearer gb_free_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Run a Node

Contribute GPU power and earn rewards.

```bash
pip install gitblock[node]

# Configure
gitblock-node --wallet 0xYourAddr --model llama-3.3-70b

# Start serving
python -m node
```

## Features

- ⚡ **Free Inference** — 80+ open-source models at zero cost
- 🔗 **OpenAI Compatible** — Same API format, swap the base URL
- 🌐 **Multi-Language** — Python SDK, JavaScript SDK, REST API, CLI
- 🔒 **Privacy First** — End-to-end encrypted, no logging
- 🏗️ **Open Source** — MIT licensed, fork and deploy your own network

## Available Models

| Model | Size | Category |
|-------|------|----------|
| Llama 3.3 | 70B | General |
| Mistral 7B | 7B | Fast |
| DeepSeek Coder | 33B | Code |
| Qwen 2.5 | 72B | General |
| Gemma 4 | 9B | Fast |
| Phi-3 | 14B | Fast |
| CodeLlama | 34B | Code |

## Architecture

```
User → API Gateway → Router → Node (GPU) → Response
              ↓
         On-chain log
         Reputation score
         Token rewards
```

## Project Structure

```
Gitblock/
├── gitblock/           # Python SDK
│   ├── __init__.py     # Package exports
│   ├── client.py       # Main API client
│   ├── models.py       # Data models
│   ├── streaming.py    # SSE stream parser
│   ├── errors.py       # Custom exceptions
│   └── cli/            # CLI tool
│       ├── main.py     # Entry point
│       ├── chat.py     # Interactive REPL
│       └── utils.py    # Helpers & config
├── node/               # Node server software
│   ├── server.py       # FastAPI server
│   ├── router.py       # Smart routing
│   ├── reputation.py   # Reputation scoring
│   ├── rewards.py      # Token rewards
│   └── config.py       # Configuration
├── index.html          # Landing page
├── pyproject.toml      # Package config
└── requirements.txt    # Dependencies
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a PR

## License

MIT — Free for everyone, forever.

## Links

- 🌐 [Website](https://gitblock.org)
- 🐙 [GitHub](https://github.com/Gitblock17/Gitblock)
- 🐦 [Twitter/X](https://x.com/gitblock_)
