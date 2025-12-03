# -*- coding: utf-8 -*-
"""
AgentScope Extension for Nacos - Deep integration between AgentScope and Nacos

This library provides seamless integration of AgentScope with Nacos, supporting:
- Dynamic Agent configuration management (Prompt, MCP Server, Chat Model)
- Nacos service discovery and registration
- MCP (Model Context Protocol) clients
- A2A (Agent-to-Agent) protocol support

Core Components:
- NacosServiceManager: Nacos service connection pool manager (Singleton pattern)
- NacosAgentListener: Agent configuration listener with Nacos integration
- NacosReActAgent: ReAct Agent integrated with Nacos configuration

Submodules:
- mcp: MCP protocol clients and dynamic toolkit
- model: Dynamically configured chat models
- a2a: A2A protocol agents and adapters

Usage Examples:
    >>> from agentscope_extension_nacos import (
    ...     NacosServiceManager,
    ...     NacosAgentListener,
    ...     NacosReActAgent,
    ... )
    >>> 
    >>> # Method 1: Using environment variables
    >>> listener = NacosAgentListener(agent_name="my_agent")
    >>> await listener.initialize()
    >>> agent = NacosReActAgent(nacos_agent_listener=listener, name="my_agent")
    >>> 
    >>> # Method 2: Manually create nacos_client_config
    >>> from v2.nacos import ClientConfigBuilder
    >>> config = (ClientConfigBuilder()
    ...     .server_address("localhost:8848")
    ...     .namespace_id("public")
    ...     .username("nacos")
    ...     .password("nacos")
    ...     .build())
    >>> listener = NacosAgentListener(
    ...     agent_name="my_agent",
    ...     nacos_client_config=config,  # Pass custom config
    ... )
    >>> await listener.initialize()
    >>> 
    >>> # Method 3: Set global config (affects all components)
    >>> NacosServiceManager.set_global_config(config)

Environment Variables:
    NACOS_SERVER_ADDRESS=localhost:8848   # Required
    NACOS_NAMESPACE_ID=public             # Required
    NACOS_ACCESS_KEY=xxx                  # Optional (Alibaba Cloud MSE)
    NACOS_SECRET_KEY=yyy                  # Optional (Alibaba Cloud MSE)
    NACOS_USERNAME=nacos                  # Optional (Local Nacos)
    NACOS_PASSWORD=nacos                  # Optional (Local Nacos)
"""

__version__ = "0.2.1"
__author__ = "AgentScope Extension Team"

# =============================================================================
# Core Components - Nacos Service Manager
# =============================================================================
from agentscope_extension_nacos.nacos_service_manager import (
    NacosServiceManager,
    # Convenience functions
    get_nacos_naming_service,
    get_nacos_config_service,
    get_nacos_ai_service,
)

# =============================================================================
# Core Components - Agent Listener and ReAct Agent
# =============================================================================
from agentscope_extension_nacos.nacos_react_agent import (
    NacosAgentListener,
    NacosReActAgent,
)

# =============================================================================
# Utilities
# =============================================================================
from agentscope_extension_nacos.utils import (
    AsyncRWLock,
    validate_agent_name,
    get_first_non_loopback_ip,
    generate_url_from_endpoint,
    random_generate_url_from_mcp_server_detail_info,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Service Manager
    "NacosServiceManager",
    "get_nacos_naming_service",
    "get_nacos_config_service",
    "get_nacos_ai_service",
    # Agent Components
    "NacosAgentListener",
    "NacosReActAgent",
    # Utilities
    "AsyncRWLock",
    "validate_agent_name",
    "get_first_non_loopback_ip",
    "generate_url_from_endpoint",
    "random_generate_url_from_mcp_server_detail_info",
]
