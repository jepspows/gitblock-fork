"""GitBlock Agent Tools — Tool registry for autonomous agents.

Tools are the actions agents can take: web search, crypto data, social posting,
blockchain queries, and notifications. Each tool is a simple async function
that the agent can call during a session.

Matches Aeon's skill tool patterns but runs on GitBlock nodes using open models.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("gitblock-agent-tools")


@dataclass
class ToolDef:
    """Definition of a tool the agent can call."""
    name: str
    description: str
    parameters: dict  # JSON Schema for parameters
    handler: Callable  # async function(params) -> str


class ToolRegistry:
    """Registry of available tools for agent sessions."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._register_defaults()

    def register(self, tool: ToolDef) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict]:
        """Return tool definitions in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def get_handler(self, name: str) -> Optional[Callable]:
        """Get a tool's handler function."""
        tool = self._tools.get(name)
        return tool.handler if tool else None

    async def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool by name with arguments. Returns result string."""
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            result = await tool.handler(arguments)
            return json.dumps(result) if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return json.dumps({"error": str(e)})

    def _register_defaults(self) -> None:
        """Register the default built-in tools."""
        self.register(ToolDef(
            name="web_search",
            description="Search the web for current information. Returns relevant results with titles and URLs.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            handler=self._web_search,
        ))

        self.register(ToolDef(
            name="get_crypto_price",
            description="Get the current price of a cryptocurrency token.",
            parameters={
                "type": "object",
                "properties": {
                    "token": {"type": "string", "description": "Token symbol or ID (e.g., bitcoin, ethereum, solana)"},
                },
                "required": ["token"],
            },
            handler=self._get_crypto_price,
        ))

        self.register(ToolDef(
            name="send_notification",
            description="Send a notification to the user via configured channels.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Notification message to send"},
                    "channel": {"type": "string", "enum": ["telegram", "discord", "email"], "description": "Channel to send to"},
                },
                "required": ["message"],
            },
            handler=self._send_notification,
        ))

        self.register(ToolDef(
            name="read_memory",
            description="Read the agent's persistent memory to recall past context, goals, and notes.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Optional key to read a specific note. Omit to read all memory."},
                },
                "required": [],
            },
            handler=self._read_memory,
        ))

        self.register(ToolDef(
            name="write_memory",
            description="Save information to the agent's persistent memory for future sessions.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key/name for this memory entry"},
                    "value": {"type": "string", "description": "Value to store"},
                },
                "required": ["key", "value"],
            },
            handler=self._write_memory,
        ))

        self.register(ToolDef(
            name="get_current_time",
            description="Get the current UTC time. Use this before scheduling or timestamping anything.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=self._get_current_time,
        ))

    # ── Tool Handlers ──

    async def _web_search(self, params: dict) -> dict:
        """Search the web using DuckDuckGo."""
        query = params.get("query", "")
        try:
            import urllib.request
            import urllib.parse

            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={"User-Agent": "GitBlock-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            results = []
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic["Text"],
                    })

            return {"results": results} if results else {"results": [], "note": "No results found"}
        except Exception as e:
            return {"error": f"Search failed: {str(e)}", "results": []}

    async def _get_crypto_price(self, params: dict) -> dict:
        """Get crypto price from CoinGecko free API."""
        token = params.get("token", "").lower()
        token_map = {
            "bitcoin": "bitcoin", "btc": "bitcoin",
            "ethereum": "ethereum", "eth": "ethereum",
            "solana": "solana", "sol": "solana",
            "gitblock": "gitblock", "gblock": "gitblock",
        }
        coingecko_id = token_map.get(token, token)

        try:
            import urllib.request
            url = (
                f"https://api.coingecko.com/api/v3/simple/price"
                f"?ids={coingecko_id}&vs_currencies=usd&include_24hr_change=true"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "GitBlock-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            if coingecko_id in data:
                price_data = data[coingecko_id]
                return {
                    "token": token,
                    "price_usd": price_data.get("usd", 0),
                    "change_24h_pct": price_data.get("usd_24h_change", 0),
                }
            return {"error": f"Token '{token}' not found on CoinGecko"}
        except Exception as e:
            return {"error": f"Price fetch failed: {str(e)}"}

    async def _send_notification(self, params: dict) -> dict:
        """Send notification — stores to .pending-notify/ for delivery."""
        message = params.get("message", "")
        channel = params.get("channel", "telegram")

        try:
            from pathlib import Path
            import time

            notify_dir = Path(".pending-notify")
            notify_dir.mkdir(exist_ok=True)

            ts = int(time.time())
            file_path = notify_dir / f"agent_{ts}.json"
            file_path.write_text(json.dumps({
                "message": message,
                "channel": channel,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }))

            return {"status": "queued", "channel": channel, "preview": message[:100]}
        except Exception as e:
            return {"error": f"Notification failed: {str(e)}"}

    async def _read_memory(self, params: dict) -> dict:
        """Read agent memory. Hook — actual implementation wired at session start."""
        key = params.get("key")
        return {"status": "memory_read_placeholder", "key": key}

    async def _write_memory(self, params: dict) -> dict:
        """Write agent memory. Hook — actual implementation wired at session start."""
        return {"status": "memory_write_placeholder", "key": params.get("key"), "value": params.get("value", "")[:50]}

    async def _get_current_time(self, params: dict) -> dict:
        """Get current UTC time."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        return {
            "utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unix": int(now.timestamp()),
            "day_of_week": now.strftime("%A"),
        }
