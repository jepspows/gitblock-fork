"""GitBlock SDK exception classes.

Provides a hierarchy of exceptions for handling API errors,
authentication issues, rate limiting, and model lookup failures.
"""

from typing import Optional


class GitBlockError(Exception):
    """Base exception for all GitBlock SDK errors.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code, if applicable.
        request_id: Unique request identifier from the API, if available.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        if self.request_id is not None:
            parts.append(f"request_id={self.request_id}")
        return " | ".join(parts)


class AuthenticationError(GitBlockError):
    """Raised when the API key is missing, invalid, or expired.

    This typically corresponds to HTTP 401 responses.
    """

    def __init__(
        self,
        message: str = "Invalid or missing API key",
        status_code: Optional[int] = 401,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message, status_code, request_id)


class RateLimitError(GitBlockError):
    """Raised when the API rate limit has been exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying, if provided by the API.

    This typically corresponds to HTTP 429 responses.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        status_code: Optional[int] = 429,
        request_id: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, status_code, request_id)


class ModelNotFoundError(GitBlockError):
    """Raised when the requested model does not exist or is unavailable.

    This typically corresponds to HTTP 404 responses on model-specific endpoints.
    """

    def __init__(
        self,
        message: str = "Model not found",
        status_code: Optional[int] = 404,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message, status_code, request_id)


class APIError(GitBlockError):
    """Raised for general API errors that don't fall into specific categories.

    Attributes:
        error_type: The error type string returned by the API, if any.
        error_code: The error code string returned by the API, if any.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        self.error_type = error_type
        self.error_code = error_code
        super().__init__(message, status_code, request_id)


_STATUS_TO_EXCEPTION = {
    401: AuthenticationError,
    404: ModelNotFoundError,
    429: RateLimitError,
}


def raise_for_status(status_code: int, body: dict, request_id: Optional[str] = None) -> None:
    """Map an HTTP status code to the appropriate SDK exception and raise it.

    Args:
        status_code: The HTTP response status code.
        body: Parsed JSON response body.
        request_id: The ``x-request-id`` header value, if present.

    Raises:
        AuthenticationError: On 401.
        RateLimitError: On 429.
        ModelNotFoundError: On 404.
        APIError: On any other non-2xx status.
    """
    error_info = body.get("error", {})
    message = error_info.get("message", f"API returned status {status_code}")

    exc_cls = _STATUS_TO_EXCEPTION.get(status_code)
    if exc_cls is not None:
        kwargs: dict = {"message": message, "status_code": status_code, "request_id": request_id}
        if exc_cls is RateLimitError:
            retry_after = body.get("retry_after")
            if retry_after is not None:
                kwargs["retry_after"] = float(retry_after)
        raise exc_cls(**kwargs)

    raise APIError(
        message=message,
        status_code=status_code,
        request_id=request_id,
        error_type=error_info.get("type"),
        error_code=error_info.get("code"),
    )
