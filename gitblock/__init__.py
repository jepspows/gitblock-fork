"""GitBlock Python SDK — Decentralised AI Inference Network.

Quick start::

    from gitblock import GitBlock

    client = GitBlock(api_key="gbk_...")

    # Simple chat completion
    response = client.chat("Explain blockchain in one sentence.")
    print(response.choices[0].message.content)

    # Streaming
    with client.chat_stream("Write a haiku about code.") as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)

    # Embeddings
    emb = client.embeddings("Hello, world!")
    print(emb.data[0].embedding[:5])
"""

from __future__ import annotations

from .client import GitBlock
from .errors import (
    APIError,
    AuthenticationError,
    GitBlockError,
    ModelNotFoundError,
    RateLimitError,
)
from .models import (
    ChatResponse,
    Choice,
    DeltaMessage,
    EmbeddingData,
    EmbeddingResponse,
    Message,
    Model,
    ModelList,
    StreamChoice,
    StreamChunk,
    Usage,
)
from .streaming import SSEStream, StreamAccumulator

__all__ = [
    # Client
    "GitBlock",
    # Errors
    "GitBlockError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "APIError",
    # Models
    "Message",
    "ChatResponse",
    "Choice",
    "DeltaMessage",
    "StreamChoice",
    "StreamChunk",
    "Usage",
    "Model",
    "ModelList",
    "EmbeddingData",
    "EmbeddingResponse",
    # Streaming
    "SSEStream",
    "StreamAccumulator",
]

__version__ = "0.1.0"
