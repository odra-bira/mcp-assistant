"""
Manual MCP client session against the application tracker server, with no model involved.

Walks through the protocol directly: handshake, tools/list, tools/call,
resources/read and prompts/get. Useful for debugging a server in isolation.

Run:  python scripts/manual_client.py
"""

import asyncio
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    # 1. Describe how to START the server: "run this Python file".
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "servers" / "jobs_db_server.py")]
    )

    # 2. Connecting launches the server as a subprocess and does the HANDSHAKE.
    async with Client(params) as client:
        print("Connected to:", client.server_info.name if client.server_info else "?")

        # 3. Ask "what can you do?"  (tools/list)
        tools = await client.list_tools()
        print("\nTOOLS this server offers:")
        for t in tools.tools:
            print(f"  - {t.name}: {t.description}")

        # 4. Call a tool  (tools/call)
        result = await client.call_tool(
            "save_application",
            {"company": "Acme AI", "title": "Python Developer", "url": "https://example.com/1"},
        )
        print("\nsave_application ->", result.content[0].text)

        result = await client.call_tool("list_applications", {})
        print("list_applications ->", result.content[0].text)

        # 5. Read a resource  (resources/read)
        res = await client.read_resource("jobs://applications")
        print("\nresource jobs://applications ->", res.contents[0].text[:120], "...")

        # 6. Get a prompt  (prompts/get)
        prompt = await client.get_prompt(
            "cover_letter", {"company": "Acme AI", "title": "Python Developer"}
        )
        print("\nprompt cover_letter ->", prompt.messages[0].content.text)


if __name__ == "__main__":
    asyncio.run(main())
