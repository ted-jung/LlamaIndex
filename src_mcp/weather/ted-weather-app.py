# =============================================================================
# Example - App
# Created: 21, Mar 2025
# Updated: 21, Mar 2025
# Writer: Ted, Jung
# Description: 
#   MCP Client which is paired with MCP Server
# =============================================================================


import asyncio
import argparse

from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI

from llama_index.core.llms import ChatMessage
from llama_index.core.agent import ReActAgent, ReActChatFormatter
from llama_index.core.agent.react.prompts import REACT_CHAT_SYSTEM_HEADER

from mcp_client import MCPClient
from mcp_adapter import MCPToolAdapter


load_dotenv()  # load environment variables from .env


SYSTEM_PROMPT = """\
You are an AI assistant.
"""


async def get_agent(adapter: MCPToolAdapter):

    tools = await adapter.list_tools()
    llm = OpenAI(model="gpt-4o-mini")

    agent = ReActAgent.from_tools(
        llm=llm,
        tools=list(tools),
        react_chat_formatter=ReActChatFormatter(
            system_header=SYSTEM_PROMPT + "\n" + REACT_CHAT_SYSTEM_HEADER,
        ),
        max_iterations=5,
        verbose=True,
    )
    return agent


# async def handle_user_message(message_content: str, agent: ReActAgent):
#     user_message = ChatMessage.from_str(role="user", content=message_content)
#     response = await agent.achat(message=user_message.content)
#     print(response.response)



async def main():

    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <path_to_server.py>")
        sys.exit(1)

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--client_type", type=str, default="sse", choices=["sse", "stdio"]
    # )
    # args = parser.parse_args()

    # if args.client_type == "sse":
    #     client = MCPClient("http://127.0.0.1:8000/sse")
    # else:
    #     client = MCPClient(
    #         "../../.venv/bin/python", ["mcp_server_weather.py", "--server_type", "stdio"]
    #     )

    client = MCPClient("../../.venv/bin/python", ["mcp_server_weather.py", "--server_type", "stdio"])
    # client = MCPClient()
    adapter = MCPToolAdapter(client)
    agent = await get_agent(adapter)

    # await handle_user_message("What's the weather in CA", agent)

    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop(agent)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())