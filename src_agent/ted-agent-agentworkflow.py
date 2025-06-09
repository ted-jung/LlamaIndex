# =============================================================================
# AgentWorkflow to create multiple agents
# Created: 22, Apr 2025
# Updated: 20, May 2025
# Writer: Ted, Jung
# Description: Booking agent(FunctionCallingAgent) with functiontool
#     1. define tools(system prompt + llm + handoff)
#     2. run the workflow
#     3. handle agentstream
# ---------
#     It is a workflow that create an agent(FunctionAgent or ReActAgent) implicitly with listing tools
#     by method"from_tools_or_functions" having LLM, system prompt, a list of tools
#     PlaywrightToolSpec       : a tool for automation of web browser
#     AgentQLBrowserToolSpec   : a tool for extracting data
#     DuckDuckGoSearchToolSpec : a tool for searching
#     Search-DuckDuckGo, Click link by Playwright, Extract info by AgentQl
#     ag_workflow.run returns handler and treat this as event stream asynchronously
#     check agentstream and print delta
# ---------
# Pros: simple code, easy to handle the action of agent
# Cons: no agent definition, better to declare agent explicitly
# =============================================================================


import os
import asyncio
import nest_asyncio



# It's(agentql) designed to give your agents web browsing superpowers
# in a structured, declarative way. (browse web as tool)
# one more:
# PlaywrightToolSpec: web browser automation capability

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

llm = OpenAI(model="gpt-4.1-nano")

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


duckduckgo_search_tool = [
    tool for tool in DuckDuckGoSearchToolSpec().to_tool_list()

    if tool.metadata.name == "duckduckgo_full_search"
]
agentql_browser_tool = AgentQLBrowserToolSpec(async_browser=async_browser)



# Workflow for multiple agents with handoffs capability
# It is required a list of tool which is defined previously
# It implicitly create an agent(either FunctionAgent or ReActAgent)

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