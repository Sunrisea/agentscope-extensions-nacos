"""
Example: Prompt Configuration Hosting with Nacos

This example demonstrates how to use NacosPromptListener to dynamically manage
prompt templates through Nacos, with support for variable rendering and hot updates.

Required Nacos Configuration:
    Group: nacos-ai-prompt
    DataId: my-assistant.json
    Content: {
        "template": "You are {{role}}, a helpful assistant specialized in {{domain}}. Please respond in {{language}}."
    }

Features:
    - Dynamic prompt template management from Nacos
    - Variable rendering with {{variable}} syntax
    - Hot updates without restart when prompt changes in Nacos
    - Automatic agent prompt synchronization
"""

import asyncio
import os

from agentscope.model import DashScopeChatModel
from agentscope_extension_nacos.utils.nacos_service_manager import NacosServiceManager
from agentscope_extension_nacos.prompt.nacos_prompt_listener import NacosPromptListener
from agentscope.agent import ReActAgent, UserAgent, UserInputBase, UserInputData
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import TextBlock
from v2.nacos import ClientConfigBuilder


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
    """Create a ReAct agent with Nacos-managed prompt template."""

    # Create Nacos prompt listener with template variables
    # Variables in the template (e.g., {{role}}, {{domain}}) will be replaced
    # with values from the args dictionary
    prompt_listener = NacosPromptListener(
        prompt_key="my-assistant",
        args={
            "role": "Jarvis",
            "domain": "programming and technology",
            "language": "Chinese",
        },
    )

    # Build agent with a placeholder prompt (will be updated by listener)
    jarvis = ReActAgent(
        name="Jarvis",
        sys_prompt="",  # Will be set by NacosPromptListener
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.getenv("DASH_SCOPE_API_KEY"),
        ),
        formatter=DashScopeChatFormatter(),
        memory=InMemoryMemory(),
    )

    # Attach agent to prompt listener
    # The listener will automatically update agent's sys_prompt when:
    # 1. Initial configuration is loaded from Nacos
    # 2. Prompt configuration changes in Nacos (hot update)
    prompt_listener.attach_agent(jarvis)

    # Initialize the listener (loads prompt from Nacos and sets up change listener)
    await prompt_listener.initialize()

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

    # Cleanup: detach agent when done
    prompt_listener.detach_agent()


if __name__ == "__main__":
    asyncio.run(creating_react_agent())
