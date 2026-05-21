"""GitBlock API client.

Provides a synchronous client that mirrors the OpenAI Python SDK interface,
making it straightforward to swap between providers.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import requests

from .errors import AuthenticationError, GitBlockError, raise_for_status
from .models import (
    ChatResponse,
    EmbeddingResponse,
    Message,
    ModelList,
    StreamChunk,
)
from .streaming import SSEStream

_DEFAULT_BASE_URL = "https://api.gitblock.io/v1"
_DEFAULT_TIMEOUT = 120  # seconds
_DEFAULT_MAX_RETRIES = 3


class GitBlock:
    """The main entry point for interacting with the GitBlock API.

    All methods follow the `OpenAI Chat Completions API
    <https://platform.openai.com/docs/api-reference/chat>`_ contract,
    so existing OpenAI-compatible code can switch to GitBlock with minimal
    changes.

    Args:
        api_key: Your GitBlock API key.  If *None*, the constructor will
            look for the ``GITBLOCK_API_KEY`` environment variable.
        base_url: Override the API base URL (useful for self-hosted
            gateways or proxies).
        timeout: Default HTTP timeout in seconds for each request.
        max_retries: How many times to automatically retry on transient
            failures (rate-limits, server errors) with exponential back-off.

    Example::

        from gitblock import GitBlock

        client = GitBlock(api_key="gbk_...")
        response = client.chat("Explain quantum computing in one paragraph.")
        print(response.choices[0].message.content)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_key = api_key or os.environ.get("GITBLOCK_API_KEY", "")
        if not self.api_key:
            raise AuthenticationError(
                "No API key provided. Pass api_key= or set the GITBLOCK_API_KEY "
                "environment variable."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "gitblock-python/0.1.0",
            }
        )

    # -- low-level helpers -------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> requests.Response:
        """Send an HTTP request with automatic retries for transient errors.

        Args:
            method: HTTP verb (``GET``, ``POST``, etc.).
            path: URL path relative to :attr:`base_url`.
            json_body: Optional JSON-serialisable request body.
            stream: If *True*, return the response immediately for streaming.

        Returns:
            The :class:`requests.Response` object.

        Raises:
            GitBlockError: On non-retryable API errors.
        """
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.request(
                    method,
                    url,
                    json=json_body,
                    stream=stream,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise GitBlockError(f"Request failed after {self.max_retries + 1} attempts: {exc}") from exc

            # Successful response.
            if response.status_code < 400:
                return response

            # Retry on 429 / 5xx.
            if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 2 ** attempt
                time.sleep(wait)
                continue

            # Non-retryable error — parse and raise.
            try:
                body = response.json()
            except ValueError:
                body = {"error": {"message": response.text}}
            raise_for_status(
                response.status_code,
                body,
                request_id=response.headers.get("x-request-id"),
            )

        # Should not reach here, but just in case.
        raise GitBlockError(f"Request failed after {self.max_retries + 1} attempts", request_id=None)

    # -- chat completions --------------------------------------------------

    def chat(
        self,
        prompt: str,
        *,
        model: str = "gitblock-7b",
        messages: Optional[List[Message]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Create a chat completion.

        If *prompt* is supplied without *messages*, a single ``user`` message
        is constructed from it.  For multi-turn conversations pass *messages*
        directly.

        Args:
            prompt: A convenience shortcut — becomes the sole user message.
            model: The model to query.
            messages: An explicit list of :class:`Message` objects.
            temperature: Sampling temperature (0 – 2).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            stop: Up to four stop sequences.
            **kwargs: Any additional parameters forwarded to the API.

        Returns:
            A :class:`ChatResponse` instance.

        Example::

            resp = client.chat("Hello!")
            print(resp.choices[0].message.content)
        """
        payload = self._build_chat_payload(
            prompt=prompt,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            stream=False,
            **kwargs,
        )
        resp = self._request("POST", "/chat/completions", json_body=payload)
        return ChatResponse.from_dict(resp.json())

    def chat_stream(
        self,
        prompt: str,
        *,
        model: str = "gitblock-7b",
        messages: Optional[List[Message]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> SSEStream:
        """Create a streaming chat completion.

        Returns an :class:`~gitblock.streaming.SSEStream` context manager
        that yields :class:`~gitblock.models.StreamChunk` objects.

        Args:
            prompt: A convenience shortcut — becomes the sole user message.
            model: The model to query.
            messages: An explicit list of :class:`Message` objects.
            temperature: Sampling temperature (0 – 2).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            stop: Up to four stop sequences.
            **kwargs: Any additional parameters forwarded to the API.

        Returns:
            An :class:`SSEStream`.

        Example::

            with client.chat_stream("Tell me a story") as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        print(delta, end="", flush=True)
        """
        payload = self._build_chat_payload(
            prompt=prompt,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            stream=True,
            **kwargs,
        )
        resp = self._request("POST", "/chat/completions", json_body=payload, stream=True)
        return SSEStream(resp)

    # -- completions (legacy) ----------------------------------------------

    def completions(
        self,
        prompt: str,
        *,
        model: str = "gitblock-7b",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Create a completion using the chat completions endpoint.

        This is a thin convenience wrapper that wraps *prompt* in a single
        ``user`` message and delegates to :meth:`chat`.

        Args:
            prompt: The text prompt.
            model: The model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Extra parameters forwarded to the API.

        Returns:
            A :class:`ChatResponse`.
        """
        return self.chat(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    # -- embeddings --------------------------------------------------------

    def embeddings(
        self,
        input: Union[str, List[str]],
        *,
        model: str = "gitblock-embedding-v1",
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Create embeddings for the given input text(s).

        Args:
            input: A single string or a list of strings to embed.
            model: The embedding model to use.
            **kwargs: Extra parameters forwarded to the API.

        Returns:
            An :class:`EmbeddingResponse`.

        Example::

            resp = client.embeddings("The quick brown fox")
            vector = resp.data[0].embedding
        """
        payload: Dict[str, Any] = {"input": input, "model": model, **kwargs}
        resp = self._request("POST", "/embeddings", json_body=payload)
        return EmbeddingResponse.from_dict(resp.json())

    # -- models ------------------------------------------------------------

    def list_models(self) -> ModelList:
        """List all available models.

        Returns:
            A :class:`ModelList` instance.

        Example::

            for model in client.list_models().data:
                print(model.id)
        """
        resp = self._request("GET", "/models")
        return ModelList.from_dict(resp.json())

    def retrieve_model(self, model_id: str) -> Dict[str, Any]:
        """Retrieve metadata for a specific model.

        Args:
            model_id: The model identifier.

        Returns:
            A dictionary of model metadata.
        """
        resp = self._request("GET", f"/models/{model_id}")
        return resp.json()

    # -- private helpers ---------------------------------------------------

    @staticmethod
    def _build_chat_payload(
        prompt: str,
        model: str,
        messages: Optional[List[Message]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: Optional[float],
        stop: Optional[Union[str, List[str]]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Assemble the JSON body for a chat completions request."""
        if messages is not None:
            msgs = [m.to_dict() for m in messages]
        else:
            msgs = [Message(role="user", content=prompt).to_dict()]

        payload: Dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        payload.update(kwargs)
        return payload

    # -- dunder helpers ----------------------------------------------------

    def __repr__(self) -> str:
        masked = self.api_key[:6] + "..." if len(self.api_key) > 6 else "***"
        return f"GitBlock(base_url={self.base_url!r}, api_key={masked!r})"
