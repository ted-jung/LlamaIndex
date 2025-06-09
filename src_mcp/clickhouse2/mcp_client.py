# =============================================================================
# MCP Client for ClickHouse
# Created: 7, Apr 2025
# Updated: 19, May 2025
# Writer: Ted Jung
# Description:
#   MCP Client using MCPToolSpec in LlamaIndex to get the list of tool
#   using new module llama_index.tools.mcp
#   1. Bring a list of tools and let agent have it.
# =============================================================================


import os
import sys
import asyncio
import logging

from platform import system
from pyexpat import model
from tabnanny import verbose
from typing import List, Optional, Dict, Any

from llama_index.core.settings import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.core.workflow import Context

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from contextlib import AsyncExitStack

from llama_index.core.llms import ChatMessage

from llama_index.tools.mcp import aget_tools_from_mcp_url
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.stdio import StdioServerParameters


from mcp_env import ch_config

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = OpenAI(model="gpt-4o-mini", request_timeout=720.0)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp-clickhouse-client")

SYSTEM_PROMPT = """\
You are an AI assistant.

To help a user, you need to know the tools first. and then answer the user's question using the tools.
"""

llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

# class ClientSession:
#     async def list_tools(self) -> List[Dict]:
#         """
#         Retrieves a list of tools from the MCP server.
#         """
#         raise NotImplementedError("This method must be implemented by a concrete client.")

#     async def call_tool(self, tool_name: str, input_data: Dict) -> Dict:
#         """
#         Calls a specific tool on the MCP server.

#         Args:
#             tool_name: The name of the tool to call.
#             input_data: A dictionary of arguments for the tool.

#         Returns:
#             The result of the tool call.
#         """
#         raise NotImplementedError("This method must be implemented by a concrete client.")


class MCPClient():
    def __init__(self, server_url: str = None):
        self.mcp_server_url = server_url
        self.connected = False
        self.session: Optional[ClientSession] = None
        self.tool_spec = None
        self.exit_stack = AsyncExitStack()


    # async def connect(self) -> ReActAgent:
    #     """Establish connection to MCP server"""
    async def connect(self) -> List[FunctionTool]:
        """Establish connection to MCP server and return list of available tools"""
        if self.connected:
            return

        try:
            # Create SSE client and get list of tools using BasicMCPClient/McpToolSpec
            # Asynchronous method to convert MCP tools to FunctionTool objects.
            # return: A list of FuntionTool objects
            mcp_client = BasicMCPClient(f"http://{self.mcp_server_url}:8000/sse")

            # self.tool_spec = McpToolSpec(client=mcp_client)
            # tools = await self.tool_spec.to_tool_list_async()
            tools = await aget_tools_from_mcp_url(
                f"http://{self.mcp_server_url}:8000/sse",
                client=mcp_client
            )
            
            if mcp_client:
                self.connected = True
            
            logger.info("Successfully connected to MCP server")

            return tools

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            raise



async def get_react_agent(tools) -> ReActAgent:
    """Create a ReActAgent with ClickHouse tools"""

    # Define tools using MCPToolSpec
    # tools = [
    #     FunctionTool.from_defaults(
    #         fn=self.session.session.get_databases,
    #         name="list_databases",
    #         description="Get a list of all databases in ClickHouse"
    #     ),
    #     FunctionTool.from_defaults(
    #         fn=mcp_client.get_tables,
    #         name="list_tables",
    #         description="Get a list of tables in a specific database. Args: database (str), like (Optional[str])"
    #     ),
    #     FunctionTool.from_defaults(
    #         fn=mcp_client.run_query,
    #         name="run_query",
    #         description="Execute a SELECT query on ClickHouse. Args: query (str)"
    #     )
    # ]

    # Create ReActAgent
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=True
    )

    return agent


async def get_function_agent(tools) -> FunctionAgent:
    """Create a FunctionAgent with ClickHouse tools"""

    # Create FunctionAgent
    agent = FunctionAgent(
        name="Agent",
        description="A Function agent that can call tools",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT
    )

    return agent


async def handle_user_message(message_content: str, agent: ReActAgent):
    user_message = ChatMessage.from_str(role="user", content=message_content)
    response = await agent.achat(message=user_message.content)
    return response.response


async def main():
    """Example usage of the MCP ClickHouse client"""
    try:
        # Create and connect client
        mcp_client = MCPClient(ch_config.get_client_config()["host"])
        mcp_tools = await mcp_client.connect()

        # Create ReActAgent
        agent = await get_react_agent(mcp_tools)

        # Be able to use FunctionAgent as well
        # agent = await get_function_agent(mcp_tools)
        # agent_context = Context.from_defaults(agent=agent)
        # agent_context = Context(agent)

        print("\nReActAgent created successfully with ClickHouse tools")

        while True:
            user_input = input("Enter your message: ")
            if user_input == "exit":
                break
            print("User: ", user_input)
            response = await handle_user_message(user_input, agent)
            print("Agent: ", response)


        # Get list of databases
        # databases = await mcp_client.get_databases()
        # print("\nAvailable databases:")
        # for db in databases:
        #     print(f"- {db}")



    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())