# =============================================================================
# AgentWorkflow to create multiple agents
# Created: 22, Apr 2025
# Updated: 22, Apr 2025
# Writer: Ted, Jung
# Description: Booking agent(FunctionCallingAgent) with functiontool
#     1. define tools
#     2. define agents (tools + system prompt + llm + handoff)
#     3. define agent workflow (agents + root agent + initial state)
#     4. run the workflow
# =============================================================================


import asyncio
import os
import nest_asyncio


from llama_index.tools.playwright.base import PlaywrightToolSpec
from llama_index.tools.agentql import AgentQLBrowserToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import (
    AgentStream,
    AgentWorkflow
)


os.environ["AGENTQL_API_KEY"]
os.environ["OPENAI_API_KEY"]

nest_asyncio.apply()


async def create_async_browser():
    return await PlaywrightToolSpec.create_async_playwright_browser(
        headless=True
    )


# Create an async browser instance
# PlaywrightToolSpec is a subclass of ToolSpec that provides a way to access url
# resources using Playwright via async browser
async_browser = asyncio.run(create_async_browser())


playwright_tool = PlaywrightToolSpec(async_browser=async_browser)
playwright_tool_list = playwright_tool.to_tool_list()
playwright_agent_tool_list = [
    tool for tool in playwright_tool_list

    if tool.metadata.name in ["click", "get_current_page", "navigate_to"]
]


llm = OpenAI(model="gpt-4o")
duckduckgo_search_tool = [
    tool for tool in DuckDuckGoSearchToolSpec().to_tool_list()

    if tool.metadata.name == "duckduckgo_full_search"
]
agentql_browser_tool = AgentQLBrowserToolSpec(async_browser=async_browser)

ag_workflow = AgentWorkflow.from_tools_or_functions(
    playwright_agent_tool_list
    + agentql_browser_tool.to_tool_list()
    + duckduckgo_search_tool,
    llm=llm,
    system_prompt="You are an expert that can do browser automation, data extraction and text summarization for finding and extracting data from research resources.",
)




async def main():
    handler = await ag_workflow.run(
    user_msg="""
    Use DuckDuckGoSearch to find URL resources on the web that are relevant to the research topic: What is the relationship between exercise and stress levels?
    Go through each resource found. For each different resource, use Playwright to click on link to the resource, then use AgentQL to extract information, including the name of the resource, author name(s), link to the resource, publishing date, journal name, volume number, issue number, and the abstract.
    Find more resources until there are two different resources that can be successfully extracted from.
    """
    )

    async for event in handler:
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())