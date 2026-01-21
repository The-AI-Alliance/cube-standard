"""Middleware for CUBE server."""

import logging
import traceback
from typing import Callable

from fastapi import Request, status
from fastapi.responses import JSONResponse

from cube.server.schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


class CUBEException(Exception):
    """Base exception for CUBE server errors."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict | None = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class TaskNotFoundException(CUBEException):
    """Exception for task not found errors."""

    def __init__(self, task_id: str):
        super().__init__(
            code="TASK_NOT_FOUND",
            message=f"Task with id '{task_id}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"task_id": task_id},
        )


class SessionNotFoundException(CUBEException):
    """Exception for session not found errors."""

    def __init__(self, session_id: str):
        super().__init__(
            code="SESSION_NOT_FOUND",
            message=f"Session with id '{session_id}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"session_id": session_id},
        )


class SessionLimitException(CUBEException):
    """Exception for exceeding session limits."""

    def __init__(self, limit: int):
        super().__init__(
            code="SESSION_LIMIT_EXCEEDED",
            message=f"Maximum concurrent sessions ({limit}) exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"limit": limit},
        )


class BenchmarkNotFoundException(CUBEException):
    """Exception for benchmark not found errors."""

    def __init__(self, message: str = "Benchmark not configured or not found"):
        super().__init__(
            code="BENCHMARK_NOT_FOUND",
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class ToolExecutionException(CUBEException):
    """Exception for tool execution errors."""

    def __init__(self, tool_name: str, error_message: str):
        super().__init__(
            code="TOOL_EXECUTION_ERROR",
            message=f"Error executing tool '{tool_name}': {error_message}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"tool_name": tool_name, "error": error_message},
        )


async def cube_exception_handler(request: Request, exc: CUBEException) -> JSONResponse:
    """Handle CUBE-specific exceptions."""
    logger.warning(
        f"CUBE exception: {exc.code} - {exc.message}",
        extra={"code": exc.code, "details": exc.details},
    )

    error_response = ErrorResponse(
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(
        f"Unexpected error: {str(exc)}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc(),
        },
    )

    error_response = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            details={"error": str(exc)},
        )
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )


async def request_logging_middleware(request: Request, call_next: Callable):
    """Log incoming requests."""
    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        },
    )

    response = await call_next(request)

    logger.info(
        f"Response: {response.status_code}",
        extra={
            "status_code": response.status_code,
            "path": request.url.path,
        },
    )

    return response
