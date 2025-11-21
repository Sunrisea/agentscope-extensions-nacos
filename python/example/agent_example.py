"""
Example: Complete Agent Hosting with Nacos

This example demonstrates how to create a fully Nacos-managed agent where all
configurations (prompt, model, MCP servers) are hosted in Nacos and support
dynamic updates.

Required Nacos Configurations:
    1. Prompt Configuration:
       Group: ai-agent-test-agent
       DataId: prompt.json
       Content: {"prompt": "You are a helpful assistant"}
    
    2. Model Configuration:
       Group: ai-agent-test-agent
       DataId: model.json
       Content: {
           "modelName": "qwen-max",
           "modelProvider": "dashscope",
           "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
           "apiKey": "sk-xxx",
           "args": {
               "enable_thinking": true,
               "temperature": 1
           }
       }
    
    3. MCP Server Configuration (Optional):
       Group: ai-agent-test-agent
       DataId: mcp-server.json
       Content: {
           "mcpServers": [
               {"mcpServerName": "nacos-mcp-tools-1"},
               {"mcpServerName": "nacos-mcp-tools-2"}
           ]
       }
       Note: MCP servers must be registered in Nacos MCP Registry first.
"""

import asyncio
import os

from agentscope_extension_nacos.nacos_react_agent import (
    NacosAgentListener,
    NacosReActAgent,
)
from agentscope_extension_nacos.nacos_service_manager import NacosServiceManager
from agentscope.agent import ReActAgent, UserAgent, UserInputBase, UserInputData
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import DashScopeChatModel
from v2.nacos import ClientConfigBuilder


async def creating_react_agent() -> None:
    """Create and run a Nacos-managed ReAct agent."""
    
    # Configure Nacos connection
    client_config = (
        ClientConfigBuilder()
        .server_address("localhost:8848")
        .namespace_id("public")
        .log_level("DEBUG")  # Set to DEBUG level for detailed logs
        .build()
    )

    # Set as global configuration for all Nacos services
    NacosServiceManager.set_global_config(client_config)

    # Create agent listener to manage Nacos configurations
    # The listener will load and monitor configurations for agent "test-agent"
    nacos_agent_listener = NacosAgentListener(agent_name="test-agent")
    await nacos_agent_listener.initialize()

    # Create a fully Nacos-managed agent
    # All configurations (prompt, model, tools) come from Nacos
    jarvis = NacosReActAgent(
        nacos_agent_listener=nacos_agent_listener,
        name="Jarvis",
    )

    # Alternative: Host an existing agent in Nacos
    # Uncomment the following code to use this approach:
    #
    # # Create a regular AgentScope agent
    # jarvis = ReActAgent(
    #     name="Jarvis",
    #     sys_prompt="You are an AI assistant",
    #     model=DashScopeChatModel(
    #         model_name="qwen-max",
    #         api_key=os.getenv("DASH_SCOPE_API_KEY"),
    #     ),
    #     formatter=DashScopeChatFormatter(),
    #     memory=InMemoryMemory(),
    # )
    #
    # # Attach to Nacos listener to enable configuration management
    # nacos_agent_listener.attach_agent(jarvis)

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
