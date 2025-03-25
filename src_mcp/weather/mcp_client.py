# =============================================================================
# Example - MCPClient
# Created: 21, Mar 2025
# Updated: 21, Mar 2025
# Writer: Ted, Jung
# Description: 
#   MCP Client which communicate with MCP Server
# =============================================================================


import logging

from typing import Optional
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from anthropic import Anthropic

from contextlib import asynccontextmanager
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from urllib.parse import urlparse

from llama_index.core.llms import ChatMessage
from llama_index.core.agent import ReActAgent

logger = logging.getLogger(__name__)


load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self, command_or_url: str, args: list[str] = [], env: dict[str, str] = {}):
        self.command_or_url = command_or_url
        self.args = args
        self.env = env
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
    # methods will go here


    # This is the way how to connect to MCPServer
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [ted_tool.name for ted_tool in tools])


    # Query processsing and handling tool calls
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": ted_tool.name,
            "description": ted_tool.description,
            "input_schema": ted_tool.inputSchema} for ted_tool in response.tools]


        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls

        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })
                
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)


    async def handle_user_message2(self, message_content: str, agent: ReActAgent):
        print("here0")
        user_message = ChatMessage.from_str(role="user", content=message_content)
        print("here1")
        response = await agent.achat(message=user_message.content)
        print("here2")

        return response.response



    # add the chat loop and cleanup functionality
    async def chat_loop(self, agent):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break

                # response = await self.process_query(query)
                response = await self.handle_user_message2(query, agent)

                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")


    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()



    async def _receive_loop(self, session: ClientSession):
        logger.info("Starting receive loop")
        async for message in session.incoming_messages:
            if isinstance(message, Exception):
                logger.error("Error: %s", message)
                continue
            logger.info("Received message from server: %s", message)

    @asynccontextmanager
    async def _run_session(self):
        if urlparse(self.command_or_url).scheme in ("http", "https"):
            async with sse_client(self.command_or_url) as streams:
                async with ClientSession(*streams) as session:
                    logger.info("Initializing session")
                    await session.initialize()
                    yield session
        else:
            server_parameters = StdioServerParameters(
                command=self.command_or_url, args=self.args, env=self.env
            )
            async with stdio_client(server_parameters) as streams:
                async with ClientSession(*streams) as session:
                    logger.info("Initializing session")
                    await session.initialize()
                    yield session


    async def call_tool(self, tool_name: str, arguments: dict):
        async with self._run_session() as session:
            return await session.call_tool(tool_name, arguments)


    async def list_tools(self):
        async with self._run_session() as session:
            return await session.list_tools()