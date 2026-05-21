"""Data models for the GitBlock SDK.

All models are implemented as dataclasses for simplicity and transparency.
They map directly to the JSON structures returned by the GitBlock API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: The role of the message author (``"system"``, ``"user"``,
            ``"assistant"``, or ``"tool"``).
        content: The text content of the message.
        name: An optional name for the participant.
    """

    role: str
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        return d


# ---------------------------------------------------------------------------
# Chat response
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """Token usage statistics returned by the API.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens consumed.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Usage:
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class Choice:
    """A single completion choice returned by the chat endpoint.

    Attributes:
        index: The index of this choice in the list.
        message: The assistant message.
        finish_reason: Why the model stopped generating (e.g. ``"stop"``,
            ``"length"``).
    """

    index: int
    message: Message
    finish_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Choice:
        msg_data = data.get("message", {})
        return cls(
            index=data.get("index", 0),
            message=Message(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content", ""),
            ),
            finish_reason=data.get("finish_reason"),
        )


@dataclass
class ChatResponse:
    """A non-streaming response from the chat completions endpoint.

    Attributes:
        id: Unique identifier for the completion.
        object: Always ``"chat.completion"``.
        created: Unix timestamp of when the completion was created.
        model: The model used for the completion.
        choices: List of completion choices.
        usage: Token usage statistics.
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChatResponse:
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=[Choice.from_dict(c) for c in data.get("choices", [])],
            usage=Usage.from_dict(data.get("usage", {})),
        )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@dataclass
class DeltaMessage:
    """A partial message chunk used during streaming.

    Attributes:
        role: Present only in the first chunk.
        content: The incremental text content.
    """

    role: Optional[str] = None
    content: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeltaMessage:
        return cls(
            role=data.get("role"),
            content=data.get("content"),
        )


@dataclass
class StreamChoice:
    """A single choice inside a streaming chunk.

    Attributes:
        index: The index of this choice.
        delta: The incremental message delta.
        finish_reason: Set on the final chunk for this choice.
    """

    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StreamChoice:
        return cls(
            index=data.get("index", 0),
            delta=DeltaMessage.from_dict(data.get("delta", {})),
            finish_reason=data.get("finish_reason"),
        )


@dataclass
class StreamChunk:
    """A single chunk in a streaming chat completion response.

    Attributes:
        id: Unique identifier for the completion.
        object: Always ``"chat.completion.chunk"``.
        created: Unix timestamp.
        model: The model generating the completion.
        choices: List of streaming choices.
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[StreamChoice]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StreamChunk:
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion.chunk"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=[StreamChoice.from_dict(c) for c in data.get("choices", [])],
        )


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """Represents an available model on the GitBlock platform.

    Attributes:
        id: The model identifier (e.g. ``"gitblock-7b"``).
        object: Always ``"model"``.
        created: Unix timestamp of when the model was registered.
        owned_by: The organisation or user that owns the model.
    """

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Model:
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "model"),
            created=data.get("created", 0),
            owned_by=data.get("owned_by", ""),
        )


@dataclass
class ModelList:
    """A paginated list of models.

    Attributes:
        object: Always ``"list"``.
        data: The list of :class:`Model` objects.
    """

    object: str = "list"
    data: List[Model] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelList:
        return cls(
            object=data.get("object", "list"),
            data=[Model.from_dict(m) for m in data.get("data", [])],
        )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingData:
    """A single embedding vector.

    Attributes:
        object: Always ``"embedding"``.
        embedding: The vector of floats.
        index: The index of this embedding in the list.
    """

    object: str
    embedding: List[float]
    index: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmbeddingData:
        return cls(
            object=data.get("object", "embedding"),
            embedding=data.get("embedding", []),
            index=data.get("index", 0),
        )


@dataclass
class EmbeddingResponse:
    """Response from the embeddings endpoint.

    Attributes:
        object: Always ``"list"``.
        data: The list of embedding vectors.
        model: The model used to generate the embeddings.
        usage: Token usage statistics.
    """

    object: str
    data: List[EmbeddingData]
    model: str
    usage: Usage

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmbeddingResponse:
        return cls(
            object=data.get("object", "list"),
            data=[EmbeddingData.from_dict(e) for e in data.get("data", [])],
            model=data.get("model", ""),
            usage=Usage.from_dict(data.get("usage", {})),
        )
