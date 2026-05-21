"""Allow running ``python -m gitblock`` to verify installation and list models.

Usage::

    python -m gitblock
    python -m gitblock --api-key gbk_...
    python -m gitblock models
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gitblock",
        description="GitBlock CLI — interact with the GitBlock API from the command line.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Your GitBlock API key (or set GITBLOCK_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the API base URL.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command")

    # `chat` sub-command
    chat_p = sub.add_parser("chat", help="Send a chat message and print the response.")
    chat_p.add_argument("prompt", nargs="+", help="The prompt text.")
    chat_p.add_argument("--model", default="gitblock-7b", help="Model to use.")
    chat_p.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate.")
    chat_p.add_argument("--stream", action="store_true", help="Stream the response.")

    # `models` sub-command
    sub.add_parser("models", help="List available models.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Show help if no sub-command given.
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Late import so --help is fast even without requests installed.
    from .client import GitBlock

    kwargs: dict = {}
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.base_url:
        kwargs["base_url"] = args.base_url

    try:
        client = GitBlock(**kwargs)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.command == "chat":
        prompt = " ".join(args.prompt)
        if args.stream:
            with client.chat_stream(prompt, model=args.model, max_tokens=args.max_tokens) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        print(delta, end="", flush=True)
            print()  # final newline
        else:
            response = client.chat(prompt, model=args.model, max_tokens=args.max_tokens)
            print(response.choices[0].message.content)

    elif args.command == "models":
        model_list = client.list_models()
        if not model_list.data:
            print("No models available.")
            return
        print(f"{'ID':<40} {'Owned by':<20} {'Created'}")
        print("-" * 80)
        for m in model_list.data:
            print(f"{m.id:<40} {m.owned_by:<20} {m.created}")


if __name__ == "__main__":
    main()
