"""
Example: A2A Agent Communication

This example demonstrates how to create an A2A (Agent-to-Agent) agent
that can communicate with remote agents using the A2A protocol.

Supported Agent Card Sources:
    1. From URL:
       agent_card_source="http://localhost:9999"
    
    2. From local file:
       agent_card_source="/agentcards/remoteAgentCard.json"
    
    3. From AgentCard object:
       card = AgentCard(...)
       agent_card_source=card
    
    4. Custom resolver (e.g., NacosA2ACardResolver):
       resolver = NacosA2ACardResolver(remote_agent_name="test-agent")
       agent_card_resolver=resolver

Features:
    - Connect to remote A2A agents
    - Automatic session state management
    - Multi-turn conversation support
"""

import asyncio

from agentscope.agent import UserAgent, UserInputBase, UserInputData
from agentscope.message import TextBlock

from agentscope_extension_nacos.a2a.a2a_agent import A2aAgent


async def creating_a2a_agent() -> None:
    """Create and use an A2A agent to communicate with a remote agent."""

    # Create A2A agent from URL
    # The URL should point to the agent's Agent Card
    # Default well-known path: /.well-known/agent.json
    jarvis = A2aAgent(
        agent_card_source="http://localhost:8090",
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
    # The A2A agent will communicate with the remote agent
    # Session state (task_id, context_id) is automatically managed
    msg = None
    msg = await user(msg)

    while True:
        msg = await jarvis(msg)
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break


if __name__ == "__main__":
    asyncio.run(creating_a2a_agent())
