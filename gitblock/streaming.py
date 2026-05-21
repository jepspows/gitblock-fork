"""Server-Sent Events (SSE) stream parser for the GitBlock SDK.

Handles the chunked transfer encoding used by the streaming chat completions
endpoint, yielding parsed :class:`~gitblock.models.StreamChunk` objects.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

import requests

from .models import StreamChunk


_SSE_DATA_PREFIX = "data: "
_SSE_DONE_MARKER = "data: [DONE]"


class SSEStream:
    """An iterator over an SSE response from the GitBlock API.

    This class consumes a :class:`requests.Response` with
    ``stream=True`` and yields :class:`StreamChunk` instances as they
    arrive.

    Args:
        response: A streaming ``requests.Response`` object.

    Example::

        with client.chat_stream("Hello") as stream:
            for chunk in stream:
                print(chunk.choices[0].delta.content, end="")
    """

    def __init__(self, response: requests.Response) -> None:
        self._response = response
        self._buffer: str = ""

    # -- context-manager support -------------------------------------------

    def __enter__(self) -> SSEStream:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP connection."""
        self._response.close()

    # -- iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over :class:`StreamChunk` objects until ``[DONE]``."""
        for raw_event in self._iter_sse_events():
            if raw_event == "[DONE]":
                return
            try:
                data = json.loads(raw_event)
            except json.JSONDecodeError:
                continue
            yield StreamChunk.from_dict(data)

    # -- internal helpers --------------------------------------------------

    def _iter_sse_events(self) -> Iterator[str]:
        """Yield raw SSE data payloads from the response stream.

        SSE events are delimited by blank lines (``\\n\\n``).  Each event
        may span multiple lines; we concatenate ``data:`` lines.
        """
        for chunk_bytes in self._response.iter_lines(chunk_size=None):
            if chunk_bytes is None:
                continue
            line = chunk_bytes.decode("utf-8", errors="replace")

            # An empty line signals the end of an event.
            if not line:
                if self._buffer:
                    yield self._buffer
                    self._buffer = ""
                continue

            if line.startswith(_SSE_DATA_PREFIX):
                payload = line[len(_SSE_DATA_PREFIX):]
                if self._buffer:
                    self._buffer += "\n" + payload
                else:
                    self._buffer = payload
            # Lines starting with ":" are SSE comments and are ignored.
            elif line.startswith(":"):
                continue

        # Flush anything remaining in the buffer.
        if self._buffer:
            yield self._buffer
            self._buffer = ""


class StreamAccumulator:
    """Helper that collects streamed chunks into a single assembled string.

    Useful when you want the convenience of streaming (progress feedback)
    but also need the full text at the end.

    Example::

        acc = StreamAccumulator()
        with client.chat_stream("Hello") as stream:
            for chunk in stream:
                acc.add(chunk)
                print(chunk.choices[0].delta.content, end="")
        full_text = acc.text
    """

    def __init__(self) -> None:
        self._parts: list[str] = []

    def add(self, chunk: StreamChunk) -> None:
        """Append the delta content from *chunk* to the accumulator."""
        for choice in chunk.choices:
            if choice.delta.content is not None:
                self._parts.append(choice.delta.content)

    @property
    def text(self) -> str:
        """Return the fully assembled text."""
        return "".join(self._parts)

    def __str__(self) -> str:
        return self.text
