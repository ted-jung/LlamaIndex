# =============================================================================
# Writer: Ted Jung
# Title: Example - MCP Server Composition (Mounting - Dynamic Linking)
# Created date: 28, Aug, 2025
# Updated date: 28, Aug, 2025
# Description: How to compose multiple MCP servers into one.
#     url https://gofastmcp.com/servers/composition
#     Dynamic linking: live link using mount() method
#     changes to subserver immediately reflected in main server
# =============================================================================


import asyncio
from fastmcp import FastMCP, Client


# Define MCP Servers (main, sub)
sub_mcp = FastMCP(name="DynamicService")


@sub_mcp.tool
def initial_tool():
    """Initial tool demonstration."""
    return "Initial Tool Exists"


# Mount subserver (synchronous operation)
mcp = FastMCP(name="MainAppLive")
mcp.mount(sub_mcp, prefix="dynamic")


# Add a tool AFTER mounting - it will be accessible through main_mcp
@sub_mcp.tool
def added_later():
    """Tool added after mounting."""
    return "Tool Added Dynamically!"



# Testing access to mounted tools
async def test_dynamic_mount():
    tools = await mcp.get_tools()
    print("Available tools:", list(tools.keys()))
    # Shows: ['dynamic_initial_tool', 'dynamic_added_later']
    
    async with Client(mcp) as client:
        result = await client.call_tool("dynamic_added_later")
        print("Result:", result.data)
        # Shows: "Tool Added Dynamically!"




if __name__ == "__main__":
    # asyncio.run(test_dynamic_mount())
    mcp.run(transport="http", host="127.0.0.1", port=8000)