"""
Example: MCP Tool Integration with Nacos

This example demonstrates how to use MCP (Model Context Protocol) tools
from Nacos MCP Registry with dynamic tool updates.

Prerequisites:
    - MCP servers must be registered in Nacos MCP Registry first
    - In this example, we use "nacos-mcp-1" and "nacos-mcp-2"

Features:
    - Automatic tool discovery from Nacos MCP Registry
    - Dynamic tool list updates without restart
    - Support for both stateless and stateful MCP clients
"""

import asyncio
import os

from agentscope.model import DashScopeChatModel
from agentscope_extension_nacos.model.nacos_chat_model import NacosChatModel
from agentscope_extension_nacos.nacos_service_manager import NacosServiceManager
from agentscope.agent import ReActAgent, UserAgent, UserInputBase, UserInputData
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import TextBlock
from v2.nacos import ClientConfigBuilder
from agentscope_extension_nacos.mcp.agentscope_nacos_mcp import (
    NacosHttpStatelessClient,
    NacosHttpStatefulClient,
)
# Use DynamicToolkit instead of Toolkit to support dynamic tool updates
from agentscope_extension_nacos.mcp.agentscope_dynamic_toolkit import DynamicToolkit


# Configure Nacos connection
client_config = (
    ClientConfigBuilder()
    .server_address("localhost:8848")
    .namespace_id("public")
    .log_level("DEBUG")  # Set to DEBUG level for detailed logs
    .build()
)

# Set as global configuration
NacosServiceManager.set_global_config(client_config)


async def creating_react_agent() -> None:
    """Create a ReAct agent with MCP tools from Nacos MCP Registry."""

    # Create MCP clients from Nacos MCP Registry
    # The MCP server names must match those registered in Nacos
    stateless_client = NacosHttpStatelessClient("nacos-mcp-1")
    stateful_client = NacosHttpStatefulClient("nacos-mcp-2")

    # Create dynamic toolkit
    # DynamicToolkit automatically syncs with Nacos when tool configurations change
    toolkit = DynamicToolkit()
    
    # Connect stateful client before registering
    await stateful_client.connect()
    
    # Register MCP clients to toolkit
    # Tools from these servers will be available to the agent
    await toolkit.register_mcp_client(stateful_client)
    await toolkit.register_mcp_client(stateless_client)

    # Build agent with MCP tools
    jarvis = ReActAgent(
        name="Jarvis",
        sys_prompt="You are an AI assistant",
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.getenv("DASH_SCOPE_API_KEY"),
        ),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,  # Agent can now use MCP tools
        memory=InMemoryMemory(),
    )

    # Custom user input handler that runs in thread pool to avoid blocking
    class ThreadedTerminalInput(UserInputBase):
        """Run input() in a thread pool to avoid blocking the event loop."""

        def __init__(self, input_hint: str = "User Input: ") -> None:
            self.input_hint = input_hint

        async def __call__(
            self, agent_id: str, agent_name: str, structured_model=None, *args, **kwargs
        ):
            loop = asyncio.get_event_loop()
            text_input = await loop.run_in_executor(None, input, self.input_hint)
            return UserInputData(
                blocks_input=[TextBlock(type="text", text=text_input)],
                structured_input=None,
            )

    # Create user agent with custom input handler
    user = UserAgent(name="user")
    user.override_instance_input_method(ThreadedTerminalInput())

    # Start conversation loop
    msg = None
    msg = await user(msg)

    while True:
        msg = await jarvis(msg)
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break


if __name__ == "__main__":
    asyncio.run(creating_react_agent())
