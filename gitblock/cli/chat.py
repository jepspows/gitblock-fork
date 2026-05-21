"""Interactive chat REPL for GitBlock with multi-turn conversation and streaming."""

import sys
from typing import List, Dict, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .utils import console, require_api_key, get_base_url, get_default_model, print_error, print_info

try:
    import httpx
except ImportError:
    httpx = None  # graceful fallback checked at runtime


# ── Streaming response helper ─────────────────────────────────────────────────

def _stream_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
) -> str:
    """Send a chat completion request with streaming; print tokens as they arrive."""
    if httpx is None:
        # Fall back to non-streaming via urllib
        return _chat_blocking(api_key, base_url, model, messages)

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": True}

    full_text = []
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
                            import json
                            delta = json.loads(data)["choices"][0]["delta"]
                            token = delta.get("content", "")
                            if token:
                                console.print(token, end="", highlight=False)
                                full_text.append(token)
                        except (KeyError, IndexError, json.JSONDecodeError):
                            continue
    except httpx.HTTPStatusError as exc:
        print_error(f"API error {exc.response.status_code}: {exc.response.text}")
        return ""
    except Exception as exc:
        print_error(f"Request failed: {exc}")
        return ""

    console.print()  # final newline
    return "".join(full_text)


def _chat_blocking(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
) -> str:
    """Non-streaming fallback using urllib (no httpx dependency)."""
    import json
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
            console.print(Markdown(text))
            return text
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode() if exc.fp else ""
        print_error(f"API error {exc.code}: {err_body}")
        return ""
    except Exception as exc:
        print_error(f"Request failed: {exc}")
        return ""


# ── REPL ───────────────────────────────────────────────────────────────────────

HELP_TEXT = """\
[bold]Chat commands:[/bold]
  /help          Show this help
  /model <name>  Switch model (e.g. /model gitblock-2)
  /system <msg>  Set system prompt
  /clear         Reset conversation history
  /history       Show conversation length
  /quit          Exit
"""


def run_chat(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """Run the interactive chat REPL."""
    api_key = api_key or require_api_key()
    base_url = get_base_url()
    current_model = model or get_default_model()
    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    console.print(Panel(
        f"[bold]GitBlock Chat[/bold]  ·  model: [accent]{current_model}[/accent]\n"
        "Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.",
        border_style="cyan",
    ))

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # ── Slash commands ─────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd, *rest = user_input.split(None, 1)
            arg = rest[0] if rest else ""
            cmd = cmd.lower()

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "/help":
                console.print(HELP_TEXT)
                continue
            elif cmd == "/clear":
                messages.clear()
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                print_info("Conversation cleared.")
                continue
            elif cmd == "/model":
                if arg:
                    current_model = arg
                    print_info(f"Switched model to [accent]{current_model}[/accent]")
                else:
                    print_info(f"Current model: [accent]{current_model}[/accent]")
                continue
            elif cmd == "/system":
                if arg:
                    # Replace or add system message
                    messages = [m for m in messages if m["role"] != "system"]
                    messages.insert(0, {"role": "system", "content": arg})
                    print_info("System prompt updated.")
                else:
                    print_warning("Usage: /system <prompt text>")
                continue
            elif cmd == "/history":
                user_turns = sum(1 for m in messages if m["role"] == "user")
                print_info(f"Conversation: {user_turns} user messages, {len(messages)} total")
                continue
            else:
                print_error(f"Unknown command: {cmd}. Type /help for options.")
                continue

        # ── Normal message ─────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        console.print("[bold magenta]GitBlock:[/bold magenta] ", end="")
        reply = _stream_chat(api_key, base_url, current_model, messages)

        if reply:
            messages.append({"role": "assistant", "content": reply})
        else:
            # Remove the user message if we got no reply
            messages.pop()


# Allow direct execution
if __name__ == "__main__":
    run_chat()
