# =============================================================================
# Writer: Ted Jung
# Title: Example - MCP Server Composition (Importing)
# Created date: 28, Aug, 2025
# Updated date: 28, Aug, 2025
# Description: One time import(static) of another MCP server into main MCP server.
# =============================================================================


from fastmcp import FastMCP
import asyncio



# Define Servers (main, subservers)
main_mcp = FastMCP(name="MainApp")
weather_mcp = FastMCP(name="WeatherService")



# Define tools and resources for each server
@weather_mcp.tool
def get_forecast(city: str) -> dict:
    """Get weather forecast."""
    return {"city": city, "forecast": "Sunny"}

@weather_mcp.resource("data://cities/supported")
def list_supported_cities() -> list[str]:
    """List cities with weather support."""
    return ["London", "Paris", "Tokyo"]



# Import subserver
async def setup():
    await main_mcp.import_server(weather_mcp, prefix="weather")




# Result: main_mcp now contains prefixed components:
# - Tool: "weather_get_forecast"
# - Resource: "data://weather/cities/supported" 

if __name__ == "__main__":
    # asyncio.run(setup())
    main_mcp.run()