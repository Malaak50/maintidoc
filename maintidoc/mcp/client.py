"""Client MCP Streamable HTTP utilisé par le graphe MaintiDoc."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


async def _call(url: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from mcp import Client

    async with Client(url) as client:
        result = await client.call_tool(name, arguments)
        structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            return structured
        content = getattr(result, "content", [])
        return {"content": [getattr(item, "text", str(item)) for item in content]}


def call_maintidoc_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    url = os.getenv("MAINTIDOC_MCP_URL", "http://127.0.0.1:8001/mcp")
    return asyncio.run(_call(url, name, arguments))
