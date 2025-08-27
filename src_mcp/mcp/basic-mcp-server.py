# =============================================================================
# Writer: Ted Jung
# Title: Basic MCP Server Example
# Created date: 27, Aug, 2025
# Updated date: 27, Aug, 2025
# Description: A MCP server expose several components such as 
#              tools, prompts, resource, etc
#     url https://gofastmcp.com/
# =============================================================================

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse


# A few examples(Basic, Tag-based, etc) to create a mcp server
# Create a basic server instance
# the name of instance is mcp
mcp = FastMCP(name="MyAssistantServer")
sub = FastMCP(name="sub")


# Only expose components tagged with "public"
# Hide components tagged as "internal" or "deprecated"  
# Combine both: show admin tools but hide deprecated ones
mcp = FastMCP(include_tags={"public"})
mcp = FastMCP(exclude_tags={"internal", "deprecated"})
mcp = FastMCP(include_tags={"admin"}, exclude_tags={"deprecated"})

# You can also add instructions for how to interact with the server
mcp_with_instructions = FastMCP(
    name="HelpfulAssistant",
    instructions="""
        This server provides data analysis tools.
        Call get_average() to analyze numerical data.
    """,
)




# --- Tools are functions that client can call to perform specific actions or 
#     access external resources.
@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

#     tag based filtering for tools
@mcp.tool(tags={"public", "utility"})
def public_tool() -> str:
    return "This tool is public"

@mcp.tool(tags={"internal", "admin"})
def admin_tool() -> str:
    return "This tool is for admins only"

# Server composition example
# sub under mcp(main)
@sub.tool
def hello(): 
    return "hi"


# --- Resources expose data sources that the client can read

@mcp.resource("data://config")
def get_config() -> dict:
    """Provides the application configuration."""
    return {"theme": "dark", "version": "1.0"}




# --- Resource Templates are parameterized resources that allow
#     the client to request specific data.

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """Retrieves a user's profile by ID."""
    # Do some data fetching here...
    # The {user_id} in the URI is extracted and passed to this function
    return {"id": user_id, "name": f"User {user_id}", "status": "active"}





# --- Prompts are reusable message templates for guiding the LLM
#     It takes input parameters in an array and returns a formatted string.

@mcp.prompt
def analyze_data(data_points: list[float]) -> str:
    """Creates a prompt asking for analysis of numerical data."""
    formatted_data = ", ".join(str(point) for point in data_points)
    return f"Please analyze these data points: {formatted_data}"





# --- Custom Routes allow you to define custom HTTP endpoints
#     Health check endpoints for monitoring
#     Simple status or info endpoints
#     Basic webhooks or callbacks
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    # do some health check logic here...
    return PlainTextResponse("OK")








# Running the MCP server
# Few example of running the server (default: stdio transport on port 9000)

if __name__ == "__main__":

    # This runs the server, defaulting to STDIO transport
    # mcp.run()
    mcp.run(transport="http")  # Health check at http://localhost:8000/health
    
    # To use a different transport, e.g., HTTP:
    # mcp.run(transport="http", host="127.0.0.1", port=9000)



    # Mount directly
    # This allows you to organize large applications into modular components or reuse existing servers.

    mcp.mount(sub, prefix="sub")