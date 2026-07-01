from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vmecdash")
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve", help="Run the standalone Dash app")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8050)
    serve_parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "serve":
        from vmecdash.dash_app.app import app

        app.run(host=args.host, port=args.port, debug=args.debug)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

