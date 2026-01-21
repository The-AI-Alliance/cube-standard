"""Entry point for running CUBE server via python -m cube."""

import sys

import uvicorn

from cube.server.config import config


def main():
    """Start the CUBE server."""
    try:
        uvicorn.run(
            "cube.server.app:app",
            host=config.host,
            port=config.port,
            reload=config.reload,
            log_level=config.log_level.lower(),
        )
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
