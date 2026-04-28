"""
Start the counter-cube benchmark server.

    Terminal 1:  uv run python server.py [--host HOST] [--port PORT]
    Terminal 2:  uv run python client.py <benchmark-url>

Why uvicorn.run directly (not make_benchmark_rpc_server)?
----------------------------------------------------------
make_benchmark_rpc_server uses a daemon thread, which is the right choice for
embedding a server inside a larger program.  Here the server IS the program,
so we call uvicorn.run directly (blocking) and let Ctrl+C shut it down cleanly.
"""

import argparse

import uvicorn
from counter_cube import CounterBenchmarkConfig

from cube.server import make_benchmark_jsonrpc_app


def main() -> None:
    parser = argparse.ArgumentParser(description="counter-cube benchmark server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    args = parser.parse_args()

    benchmark = CounterBenchmarkConfig().make()
    app = make_benchmark_jsonrpc_app(benchmark)

    url = f"http://{args.host}:{args.port}"
    print(f"Benchmark server running at {url}")
    print("Press Ctrl+C to stop.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
