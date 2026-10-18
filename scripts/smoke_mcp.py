#!/usr/bin/env python
"""
Protocol smoke test: talks to the server over real MCP stdio, not by importing it.

demo_agent.py calls the tool functions directly, which proves the ML works but
says nothing about the transport. This proves an actual MCP client can discover
the tools, call them and read structured results back.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


def text_of(res) -> str:
    for c in getattr(res, "content", []) or []:
        if getattr(c, "type", None) == "text":
            return c.text
    return json.dumps(getattr(res, "structuredContent", None), default=str)


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "server.py")], env=None)
    failures = 0
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"server: {init.server_info.name} v{init.server_info.version}")
            print(f"instructions attached: {bool(init.instructions)} "
                  f"({len(init.instructions or '')} chars)\n")

            tools = (await session.list_tools()).tools
            print(f"{len(tools)} tools discovered over the protocol:")
            for t in tools:
                required = t.input_schema.get("required", [])
                print(f"  {t.name:24} required={required}")
            if len(tools) != 12:
                print(f"  !! expected 12 tools, got {len(tools)}")
                failures += 1

            print("\ncalls:")
            for name, args, expect in [
                ("list_datasets", {}, "synthetic_churn"),
                ("load_dataset", {"source": "synthetic_churn"}, '"task": "binary"'),
                ("search_knowledge", {"query": "which metric for imbalanced classes", "top_k": 1}, "pr_auc"),
                ("get_recipe_schema", {}, "minimal_example"),
                ("validate_recipe_dry_run",
                 {"recipe": {"model": {"type": "lightgm"}}, "dataset_id": "synthetic_churn"}, "lightgbm"),
                ("run_experiment",
                 {"dataset_id": "synthetic_churn", "recipe": {"name": "smoke", "model": {"type": "dummy"}}},
                 '"status": "ok"'),
                ("run_experiment",
                 {"dataset_id": "synthetic_churn",
                  "recipe": {"name": "leaky", "model": {"type": "lightgbm"}}}, "LEAK_NOT_DROPPED"),
                ("leaderboard", {"dataset_id": "synthetic_churn"}, "baseline_score"),
            ]:
                out = text_of(await session.call_tool(name, args))
                ok = expect in out
                failures += 0 if ok else 1
                print(f"  {'OK  ' if ok else 'FAIL'} {name:24} -> {out[:110].replace(chr(10), ' ')}")

    print(f"\n{'all protocol checks passed' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
