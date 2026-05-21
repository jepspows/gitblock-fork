"""Helper functions for GitBlock CLI: config management, output formatting, spinners."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.theme import Theme

# ── Paths ──────────────────────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".gitblock"
CONFIG_FILE = CONFIG_DIR / "config.json"

# ── Rich console with custom theme ────────────────────────────────────────────

THEME = Theme({
    "info":    "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error":   "bold red",
    "accent":  "bold magenta",
})

console = Console(theme=THEME)

# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    """Load config from ~/.gitblock/config.json, returning {} on any error."""
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_config(cfg: Dict[str, Any]) -> None:
    """Persist config to ~/.gitblock/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def get_api_key() -> Optional[str]:
    """Return API key from env var or config file, env takes precedence."""
    return os.environ.get("GITBLOCK_API_KEY") or load_config().get("api_key")


def require_api_key() -> str:
    """Return API key or exit with a helpful message."""
    key = get_api_key()
    if not key:
        console.print(
            "[error]No API key configured.[/error]\n"
            "Set it via:\n"
            "  [accent]gitblock auth --key <YOUR_KEY>[/accent]\n"
            "  or export [accent]GITBLOCK_API_KEY=<YOUR_KEY>[/accent]"
        )
        sys.exit(1)
    return key


def get_base_url() -> str:
    """Return the API base URL (env > config > default)."""
    return (
        os.environ.get("GITBLOCK_API_URL")
        or load_config().get("base_url")
        or "https://api.gitblock.io/v1"
    )


def get_default_model() -> str:
    """Return the default model name."""
    return load_config().get("default_model") or "gitblock-1"


# ── Output helpers ─────────────────────────────────────────────────────────────

def print_banner() -> None:
    """Print the GitBlock ASCII banner."""
    banner = r"""
  ____ _ _   _     _    ____  _  __  __
 / ___(_) |_(_)___| |  | __ )| |/ / |  _ \
| |  _| | __| / __| |  |  _ \| ' /  | |_) |
| |_| | | |_| \__ \ |__| |_) | . \  |  _ <
 \____|_|\__|_|___/____|____/|_|\_\ |_| \_\
"""
    console.print(f"[accent]{banner}[/accent]")


def print_error(msg: str) -> None:
    console.print(f"[error]✗ {msg}[/error]")


def print_success(msg: str) -> None:
    console.print(f"[success]✓ {msg}[/success]")


def print_info(msg: str) -> None:
    console.print(f"[info]{msg}[/info]")


def print_warning(msg: str) -> None:
    console.print(f"[warning]⚠ {msg}[/warning]")


def format_json(data: Any) -> None:
    """Pretty-print JSON data."""
    from rich.json import JSON
    console.print(JSON.from_data(data))


def make_spinner(text: str = "Thinking...") -> Progress:
    """Return a Rich progress context manager with a spinner."""
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )


def print_table(title: str, columns: list, rows: list) -> None:
    """Render a Rich table."""
    table = Table(title=title, show_lines=True)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*[str(c) for c in row])
    console.print(table)
