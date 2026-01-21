"""Configuration management for CUBE server."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    """Server configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CUBE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Auto-reload on code changes (dev only)")

    # Environment
    environment: Literal["development", "production"] = Field(
        default="development", description="Runtime environment"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    # CORS settings
    cors_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    cors_credentials: bool = Field(default=True, description="Allow credentials")
    cors_methods: list[str] = Field(default=["*"], description="Allowed HTTP methods")
    cors_headers: list[str] = Field(default=["*"], description="Allowed headers")

    # Session settings
    session_timeout_seconds: int = Field(
        default=3600, description="Session timeout in seconds (1 hour)"
    )
    max_concurrent_sessions: int = Field(
        default=100, description="Maximum concurrent sessions"
    )

    # Benchmark settings
    benchmark_module: str | None = Field(
        default=None, description="Python module path to benchmark (e.g., 'examples.simple_math.benchmark')"
    )
    benchmark_class: str | None = Field(
        default=None, description="Benchmark class name (e.g., 'SimpleMathBenchmark')"
    )

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


# Global config instance
config = ServerConfig()
