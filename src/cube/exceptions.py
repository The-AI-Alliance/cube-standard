"""
CUBE Standard Exception Types.

These exceptions represent errors defined in the CUBE specification.
They can be raised by any CUBE implementation (server or client).
"""

from typing import Any


class CUBEException(Exception):
    """
    Base exception for CUBE protocol errors.

    All CUBE-compliant implementations should use these exception types
    to represent standard error conditions defined in the specification.

    Attributes:
        code: Error code (e.g., "TASK_NOT_FOUND")
        message: Human-readable error message
        details: Additional error details as a dictionary
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


class TaskNotFoundException(CUBEException):
    """Task with specified ID was not found."""

    def __init__(self, task_id: str):
        super().__init__(
            code="TASK_NOT_FOUND",
            message=f"Task with id '{task_id}' not found",
            details={"task_id": task_id},
        )


class SessionNotFoundException(CUBEException):
    """Session with specified ID was not found."""

    def __init__(self, session_id: str):
        super().__init__(
            code="SESSION_NOT_FOUND",
            message=f"Session with id '{session_id}' not found",
            details={"session_id": session_id},
        )


class SessionLimitException(CUBEException):
    """Maximum concurrent sessions exceeded."""

    def __init__(self, limit: int):
        super().__init__(
            code="SESSION_LIMIT_EXCEEDED",
            message=f"Maximum concurrent sessions ({limit}) exceeded",
            details={"limit": limit},
        )


class BenchmarkNotFoundException(CUBEException):
    """Benchmark not found or not configured."""

    def __init__(self, message: str = "Benchmark not configured or not found"):
        super().__init__(
            code="BENCHMARK_NOT_FOUND",
            message=message,
        )


class ToolExecutionException(CUBEException):
    """Error occurred during tool execution."""

    def __init__(self, tool_name: str, error_message: str):
        super().__init__(
            code="TOOL_EXECUTION_ERROR",
            message=f"Error executing tool '{tool_name}': {error_message}",
            details={"tool_name": tool_name, "error": error_message},
        )


class InvalidActionException(CUBEException):
    """Action is not valid for the current task."""

    def __init__(self, action_name: str, reason: str):
        super().__init__(
            code="INVALID_ACTION",
            message=f"Action '{action_name}' is invalid: {reason}",
            details={"action_name": action_name, "reason": reason},
        )


class TaskClosedException(CUBEException):
    """Attempted operation on a closed task session."""

    def __init__(self, session_id: str):
        super().__init__(
            code="TASK_CLOSED",
            message=f"Task session '{session_id}' is closed",
            details={"session_id": session_id},
        )


class ResourceNotFoundException(CUBEException):
    """Requested resource URI was not found."""

    def __init__(self, uri: str):
        super().__init__(
            code="RESOURCE_NOT_FOUND",
            message=f"Resource not found: {uri}",
            details={"uri": uri},
        )
