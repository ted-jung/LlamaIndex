# =============================================================================
# Writer: Ted Jung
# Title: Basic MCP Client Example
# Created date: 27, Aug, 2025
# Updated date: 27, Aug, 2025
# Description: A MCP Client to connect to a MCP server expose several components such as
# =============================================================================


import asyncio
from fastmcp import Client

client = Client("basic-mcp-server.py")

async def call_tool(name: str):
    async with client:
        result = await client.call_tool("greet", {"name": name})
        print(result)

asyncio.run(call_tool("Ford"))