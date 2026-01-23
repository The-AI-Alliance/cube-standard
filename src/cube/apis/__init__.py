from .benchmark import *
from .task import *

# =============================================================================
# Common Schemas
# =============================================================================

class ErrorDetail(BaseModel):
    """Standard error response."""

    code: str = Field(..., description="Error code (e.g., TASK_NOT_FOUND)")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")

class ErrorResponse(BaseModel):
    """Error response wrapper."""
    error: ErrorDetail

# =============================================================================
# Health Check
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Health status")
    version: str = Field(..., description="Server version")
    environment: str = Field(..., description="Runtime environment")
