# =============================================================================
# Writer: Ted Jung
# Title: Example - MCP Client
# Created date: 28, Aug, 2025
# Updated date: 28, Aug, 2025
# Description: Calling tools from MCP server
# =============================================================================


import asyncio
from fastmcp import Client

async def main():
    try:
        # Use a context manager for the client to ensure the connection is handled correctly.
        # The URL should match where your server is running.
        async with Client("http://127.0.0.1:8000/mcp/") as client:
            print("Client connected to server.")

            # get_tools() returns a dictionary of tools
            tools = await client.list_tools()

            print("\nAvailable tools:")
            for tool_name in tools:
                print(f"- {tool_name}")


            result = await client.call_tool("dynamic_added_later")
            print("======")
            print(result.content[0].text)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())