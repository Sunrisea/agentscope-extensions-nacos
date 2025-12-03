# -*- coding: utf-8 -*-
"""
A2A (Agent-to-Agent) Module

Provides A2A protocol support enabling AgentScope Agents to communicate
with other A2A-compatible agents:
- Agent Card resolvers (URL, file, Nacos)
- A2A protocol Agent wrapper
- Message format conversion

Main Classes:
- A2ACardResolverBase: Base class for Agent Card resolvers
- DefaultA2ACardResolver: Default resolver supporting URL and file paths
- A2aAgent: A2A protocol Agent wrapping remote A2A service calls

Submodules:
- nacos: Nacos-integrated A2A functionality (adapters and resolvers)

Usage Examples:
    >>> from agentscope_extension_nacos.a2a import (
    ...     A2aAgent,
    ...     DefaultA2ACardResolver,
    ... )
    >>> 
    >>> # Method 1: Create Agent from URL
    >>> agent = A2aAgent(agent_card_source="https://example.com/agent-card")
    >>> response = await agent.reply(msg)
    >>> 
    >>> # Method 2: Use custom resolver
    >>> resolver = DefaultA2ACardResolver(
    ...     agent_card_source="path/to/agent-card.json",
    ...     httpx_client=client,
    ... )
    >>> agent = A2aAgent(agent_card_source=None, agent_card_resolver=resolver)
"""

from agentscope_extension_nacos.a2a.a2a_agent import (
    A2ACardResolverBase,
    DefaultA2ACardResolver,
    A2aAgent,
)

__all__ = [
    # Base classes and resolvers
    "A2ACardResolverBase",
    "DefaultA2ACardResolver",
    # Agent
    "A2aAgent",
]
