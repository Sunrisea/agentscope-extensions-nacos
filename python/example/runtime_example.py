"""
Example: Deploy NacosReActAgent with AgentScope Runtime

This example demonstrates how to deploy a Nacos-managed agent using
AgentScope Runtime with A2A protocol support.

Required Nacos Configurations:
    See agent_example.py for required configurations (prompt.json, model.json, etc.)

Features:
    - Deploy agent as a service with AgentScope Runtime
    - A2A protocol support for agent-to-agent communication
    - Agent Card registration to Nacos A2A Registry
    - Streaming response support
"""

import asyncio
from contextlib import asynccontextmanager

from agentscope_runtime.engine import Runner, LocalDeployManager
from agentscope_runtime.engine.agents.agentscope_agent import AgentScopeAgent
from agentscope_runtime.engine.services.context_manager import ContextManager
from v2.nacos import ClientConfigBuilder
from agentscope_extension_nacos.a2a.nacos.nacos_a2a_adapter import (
    A2AFastAPINacosAdaptor,
)
from agentscope_extension_nacos.nacos_react_agent import (
    NacosReActAgent,
    NacosAgentListener,
)

# Configure Nacos connection
client_config = (
    ClientConfigBuilder()
    .server_address("localhost:8848")
    .namespace_id("public")
    .log_level("DEBUG")  # Set to DEBUG level for detailed logs
    .build()
)

# Create Nacos agent listener
# This will load agent configurations from Nacos
nacos_agent_listener = NacosAgentListener(
    nacos_client_config=client_config, agent_name="test-agent"
)

agent: AgentScopeAgent | None = None

print("✅ AgentScope agent created successfully")


@asynccontextmanager
async def create_runner():
    """Create and initialize the agent runner."""
    global agent
    
    # Initialize Nacos listener to load configurations
    await nacos_agent_listener.initialize()
    
    # Create AgentScope agent with Nacos-managed configurations
    agent = AgentScopeAgent(
        name="Friday",
        model=nacos_agent_listener.chat_model,
        agent_config={
            "nacos_agent_listener": nacos_agent_listener,
        },
        agent_builder=NacosReActAgent,
    )

    # Create runner with context manager
    async with Runner(
        agent=agent,
        context_manager=ContextManager(),
    ) as runner:
        print("✅ Runner created successfully")
        yield runner


async def deploy_agent(runner):
    """Deploy the agent as a service."""
    
    # Create deployment manager
    # This will serve the agent on localhost:8090
    deploy_manager = LocalDeployManager(
        host="localhost",
        port=8090,
    )

    # Create A2A protocol adapter
    # This exposes the agent via A2A protocol and registers it to Nacos
    a2a_protocol = A2AFastAPINacosAdaptor(
        nacos_client_config=client_config,
        agent=agent,
        host="localhost",
    )
    
    # Deploy agent with A2A protocol support
    deploy_result = await runner.deploy(
        deploy_manager=deploy_manager,
        endpoint_path="/process",
        stream=True,  # Enable streaming responses
    )
    
    print(f"🚀 Agent deployed at: {deploy_result}")
    print(f"🌐 Service URL: {deploy_manager.service_url}")
    print(f"💚 Health check: {deploy_manager.service_url}/health")

    return deploy_manager


async def run_deployment():
    """Run the deployment and keep the service alive."""
    async with create_runner() as runner:
        deploy_manager = await deploy_agent(runner)

    # Keep the service running
    # In production, you'd handle this differently (e.g., with proper shutdown handlers)
    print("🏃 Service is running...")
    await asyncio.sleep(1000)

    return deploy_manager


if __name__ == "__main__":
    asyncio.run(run_deployment())
