"""
Example: Model Configuration Hosting with Nacos

This example demonstrates how to use NacosChatModel to dynamically manage
model configurations through Nacos, including model provider, API keys, and
invocation parameters.

Required Nacos Configuration:
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

Supported Model Providers:
    - openai: OpenAI GPT series
    - anthropic: Anthropic Claude series
    - ollama: Ollama local models
    - gemini: Google Gemini
    - dashscope: Alibaba Cloud Qwen
"""

import asyncio
import os

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
    """Create a ReAct agent with Nacos-managed model and MCP tools."""

    # Create MCP clients from Nacos MCP Registry
    # Stateless client: suitable for low-frequency calls
    stateless_client = NacosHttpStatelessClient("nacos-mcp-1")
    # Stateful client: suitable for high-frequency calls
    stateful_client = NacosHttpStatefulClient("nacos-mcp-2")

    # Create dynamic toolkit that auto-syncs with Nacos
    toolkit = DynamicToolkit()
    await stateful_client.connect()
    await toolkit.register_mcp_client(stateful_client)
    await toolkit.register_mcp_client(stateless_client)

    # Create Nacos-managed chat model
    # Model configuration will be loaded from Nacos and supports hot updates
    # The model will automatically switch when configuration changes in Nacos
    model = NacosChatModel(
        agent_name="test-agent",  # Corresponds to Group: ai-agent-test-agent
        stream=True,
    )

    # Build agent with Nacos-managed model and dynamic toolkit
    jarvis = ReActAgent(
        name="Jarvis",
        sys_prompt="You are an AI assistant",
        model=model,
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
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
