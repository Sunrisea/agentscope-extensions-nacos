"""
Example: Deploy Agent as A2A Service with Nacos Registration

This example demonstrates how to deploy an agent as an A2A service
and automatically register it to Nacos A2A Registry.

Deployment Effects:
    - Agent is exposed via FastAPI with A2A protocol
    - Agent Card is automatically registered to Nacos A2A Registry
    - Other clients can discover and connect to this agent via Nacos
    - Supports streaming responses and full A2A protocol features

Client Access:
    After deployment, other clients can discover and use this agent using:
    - NacosA2ACardResolver (see nacos_a2a_example.py)
    - Direct URL access to /.well-known/agent.json
"""

import asyncio
import os
from contextlib import asynccontextmanager

from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
from agentscope_runtime.engine import Runner, LocalDeployManager
from agentscope_runtime.engine.agents.agentscope_agent import AgentScopeAgent
from agentscope_runtime.engine.services.context_manager import ContextManager
from v2.nacos import ClientConfigBuilder

from agentscope_extension_nacos.a2a.nacos import (
    A2AFastAPINacosAdaptor,
)

# Configure Nacos connection
client_config = (
    ClientConfigBuilder()
    .server_address("localhost:8848")
    .namespace_id("public")
    .log_level("DEBUG")  # Set to DEBUG level for detailed logs
    .build()
)

agent: AgentScopeAgent | None = None

print("✅ AgentScope agent created successfully")


@asynccontextmanager
async def create_runner():
    """Create and initialize the agent runner."""
    global agent

    # Create AgentScope agent
    agent = AgentScopeAgent(
        name="Friday",
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.getenv("DASH_SCOPE_API_KEY"),
        ),
        agent_config={
            "sys_prompt": "You're a helpful assistant named Friday.",
        },
        agent_builder=ReActAgent,
    )
    
    # Create runner with context manager
    async with Runner(
        agent=agent,
        context_manager=ContextManager(),
    ) as runner:
        print("✅ Runner created successfully")
        yield runner


async def deploy_agent(runner):
    """Deploy the agent as an A2A service and register to Nacos."""
    
    # Create deployment manager
    # This will serve the agent on localhost:8090
    deploy_manager = LocalDeployManager(
        host="localhost",
        port=8090,
    )
    
    # Create A2A Nacos adapter
    # This will:
    # 1. Expose the agent via A2A protocol
    # 2. Register Agent Card to Nacos A2A Registry
    # 3. Enable other clients to discover this agent via Nacos
    nacos_a2a_protocol = A2AFastAPINacosAdaptor(
        nacos_client_config=client_config,
        agent=agent,
        host="localhost",
    )
    
    # Deploy agent with A2A protocol adapter
    deploy_result = await runner.deploy(
        deploy_manager=deploy_manager,
        endpoint_path="/process",
        protocol_adapters=[nacos_a2a_protocol],  # Enable A2A protocol
        stream=True,  # Enable streaming responses
    )
    
    print(f"🚀 Agent deployed at: {deploy_result}")
    print(f"🌐 Service URL: {deploy_manager.service_url}")
    print(f"💚 Health check: {deploy_manager.service_url}/health")
    print(f"📝 Agent Card registered to Nacos A2A Registry")

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
