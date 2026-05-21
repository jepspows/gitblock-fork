"""GitBlock CLI entry point — argparse + rich powered."""

import argparse
import json
import sys
from typing import Optional

from .utils import (
    console,
    load_config,
    save_config,
    get_api_key,
    get_base_url,
    get_default_model,
    require_api_key,
    print_banner,
    print_error,
    print_success,
    print_info,
    print_warning,
    print_table,
    format_json,
    make_spinner,
    CONFIG_FILE,
)

# ── Version ────────────────────────────────────────────────────────────────────

try:
    from gitblock import __version__
except ImportError:
    __version__ = "0.1.0"


# ── Sub-command implementations ───────────────────────────────────────────────

def cmd_ask(args: argparse.Namespace) -> None:
    """One-shot question to the GitBlock API."""
    api_key = require_api_key()
    base_url = get_base_url()
    model = args.model or get_default_model()
    question = " ".join(args.question)

    if not question:
        # Read from stdin if piped
        if not sys.stdin.isatty():
            question = sys.stdin.read().strip()
        if not question:
            print_error("No question provided. Usage: gitblock ask \"your question\"")
            sys.exit(1)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": question})

    # Use streaming if httpx available, else blocking
    try:
        import httpx
        _do_streaming_request(api_key, base_url, model, messages)
    except ImportError:
        _do_blocking_request(api_key, base_url, model, messages)


def cmd_models(args: argparse.Namespace) -> None:
    """List available models."""
    api_key = require_api_key()
    base_url = get_base_url()

    try:
        import httpx
        url = f"{base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = httpx.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        models = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(models, list):
            rows = []
            for m in models:
                mid = m.get("id", m) if isinstance(m, dict) else str(m)
                owner = m.get("owned_by", "—") if isinstance(m, dict) else "—"
                rows.append([mid, owner])
            print_table("Available Models", ["Model ID", "Owner"], rows)
        else:
            format_json(data)
    except ImportError:
        print_error("Install httpx for API calls: pip install httpx")
        sys.exit(1)
    except Exception as exc:
        print_error(f"Failed to fetch models: {exc}")
        sys.exit(1)


def cmd_auth(args: argparse.Namespace) -> None:
    """Set or show the API key."""
    if args.show:
        key = get_api_key()
        if key:
            masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "****"
            print_info(f"API key: {masked}")
            print_info(f"Source:  {'env var' if __import__('os').environ.get('GITBLOCK_API_KEY') else str(CONFIG_FILE)}")
        else:
            print_warning("No API key configured.")
        return

    if args.key:
        cfg = load_config()
        cfg["api_key"] = args.key
        save_config(cfg)
        print_success(f"API key saved to {CONFIG_FILE}")
    elif args.remove:
        cfg = load_config()
        cfg.pop("api_key", None)
        save_config(cfg)
        print_success("API key removed from config.")
    else:
        console.print(
            "[info]Usage:[/info]\n"
            "  gitblock auth --key <YOUR_KEY>    Set API key\n"
            "  gitblock auth --show               Show current key\n"
            "  gitblock auth --remove             Remove saved key\n"
            "\nYou can also set [accent]GITBLOCK_API_KEY[/accent] env var."
        )


def cmd_config(args: argparse.Namespace) -> None:
    """Show or modify configuration."""
    cfg = load_config()

    if args.set:
        key, _, value = args.set.partition("=")
        if not key or not value:
            print_error("Format: --set key=value (e.g. --set default_model=gitblock-2)")
            sys.exit(1)
        # Support dotted keys like base_url
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass  # keep as string
        cfg[key.strip()] = value
        save_config(cfg)
        print_success(f"Set {key.strip()} = {value}")
        return

    if args.get:
        val = cfg.get(args.get)
        if val is not None:
            console.print(f"{args.get} = {val}")
        else:
            print_warning(f"Key '{args.get}' not set.")
        return

    # Default: show all config (mask API key)
    display = dict(cfg)
    if "api_key" in display:
        k = display["api_key"]
        display["api_key"] = k[:8] + "..." + k[-4:] if len(k) > 12 else "****"
    print_info(f"Config file: {CONFIG_FILE}")
    format_json(display)


def cmd_chat(args: argparse.Namespace) -> None:
    """Launch interactive chat REPL."""
    from .chat import run_chat
    run_chat(
        api_key=get_api_key(),
        model=args.model,
        system_prompt=args.system,
    )


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _do_streaming_request(api_key: str, base_url: str, model: str, messages: list) -> None:
    """Stream tokens from the chat completions endpoint."""
    import httpx

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": True}

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                buffer = ""
                for chunk in resp.iter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"]
                            token = delta.get("content", "")
                            if token:
                                console.print(token, end="", highlight=False)
                        except (KeyError, IndexError, json.JSONDecodeError):
                            continue
        console.print()
    except httpx.HTTPStatusError as exc:
        print_error(f"API error {exc.response.status_code}: {exc.response.text}")
        sys.exit(1)
    except Exception as exc:
        print_error(f"Request failed: {exc}")
        sys.exit(1)


def _do_blocking_request(api_key: str, base_url: str, model: str, messages: list) -> None:
    """Non-streaming request via urllib."""
    import urllib.request
    import urllib.error

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = json.dumps({"model": model, "messages": messages}).encode()

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            text = data["choices"][0]["message"]["content"]
            console.print(text)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode() if exc.fp else ""
        print_error(f"API error {exc.code}: {err_body}")
        sys.exit(1)
    except Exception as exc:
        print_error(f"Request failed: {exc}")
        sys.exit(1)


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gitblock",
        description="GitBlock CLI — Decentralized AI Inference Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  gitblock ask \"What is a transformer model?\"\n"
               "  gitblock chat --model gitblock-2\n"
               "  gitblock models\n"
               "  gitblock auth --key gbk_abc123\n",
    )
    parser.add_argument("-V", "--version", action="version", version=f"gitblock {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── ask ─────────────────────────────────────────────────────────────────
    p_ask = sub.add_parser("ask", help="Ask a one-shot question")
    p_ask.add_argument("question", nargs="*", help="The question to ask")
    p_ask.add_argument("-m", "--model", help="Model to use")
    p_ask.add_argument("-s", "--system", help="System prompt")
    p_ask.set_defaults(func=cmd_ask)

    # ── chat ────────────────────────────────────────────────────────────────
    p_chat = sub.add_parser("chat", help="Interactive chat REPL")
    p_chat.add_argument("-m", "--model", help="Model to use")
    p_chat.add_argument("-s", "--system", help="System prompt")
    p_chat.set_defaults(func=cmd_chat)

    # ── models ──────────────────────────────────────────────────────────────
    p_models = sub.add_parser("models", help="List available models")
    p_models.set_defaults(func=cmd_models)

    # ── auth ────────────────────────────────────────────────────────────────
    p_auth = sub.add_parser("auth", help="Manage API key")
    p_auth.add_argument("--key", help="Set API key")
    p_auth.add_argument("--show", action="store_true", help="Show current key")
    p_auth.add_argument("--remove", action="store_true", help="Remove saved key")
    p_auth.set_defaults(func=cmd_auth)

    # ── config ──────────────────────────────────────────────────────────────
    p_config = sub.add_parser("config", help="Show or modify configuration")
    p_config.add_argument("--set", metavar="KEY=VALUE", help="Set a config value")
    p_config.add_argument("--get", metavar="KEY", help="Get a config value")
    p_config.set_defaults(func=cmd_config)

    return parser


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        print_banner()
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
