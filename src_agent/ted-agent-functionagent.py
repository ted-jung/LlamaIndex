# =============================================================================
# Simple FunctionAgent
# Created: 20, May 2025
# Updated: 20, May 2025
# Writer: Ted, Jung
# Description: FunctionAgent
# =============================================================================


import asyncio

from llama_index.core.agent.workflow import (
    FunctionAgent, 
    AgentStream,
    AgentOutput
)

from llama_index.tools.playwright import PlaywrightToolSpec
from llama_index.llms.openai import OpenAI



llm = OpenAI(model="gpt-4.1-nano")

# 1. Create the full list of Playwright tools
playwright_tool_spec = PlaywrightToolSpec()
playwright_tool_list = playwright_tool_spec.to_tool_list()

playwright_agent_tool_list = [
    tool for tool in playwright_tool_list
    if tool.metadata.name in ["click", "get_current_page", "navigate_to"]
]


# 3. Build the agent with all tools
agent = FunctionAgent(tools=playwright_agent_tool_list, llm=llm)


async def main():
    handler = await agent.run("Go to https://www.donga.com/news/Economy/article/all/20250520/131645198/1 and then extract all of the text from the page and tell me what the page says.")

    print(handler.response)

    # async for event in handler:
    #     if isinstance(event, AgentStream):
    #         print(event.delta, end="", flush=True)


# 5. Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(main())

