# -*- coding: utf-8 -*-
"""
AgentScope Extension for Nacos - Deep integration between AgentScope and Nacos

This library provides seamless integration of AgentScope with Nacos, supporting:
- Dynamic Agent configuration management (Prompt, Model Config)
- Nacos service discovery and registration
- MCP (Model Context Protocol) clients and dynamic toolkit

Core Components:
- NacosServiceManager: Nacos service connection pool manager (Singleton pattern)
- NacosChatModel: Dynamically configured chat model from Nacos
- NacosPromptListener: Prompt template listener with variable rendering
- DynamicToolkit: MCP toolkit with automatic tool synchronization

Submodules:
- mcp: MCP protocol clients and dynamic toolkit
- model: Dynamically configured chat models
- prompt: Prompt template management with hot updates

Usage Examples:
    >>> from agentscope_extension_nacos import NacosServiceManager
    >>> from agentscope_extension_nacos.model import NacosChatModel
    >>> from agentscope_extension_nacos.prompt import NacosPromptListener
    >>> 
    >>> # Method 1: Using environment variables
    >>> model = NacosChatModel(agent_name="my_agent")
    >>> 
    >>> # Method 2: Manually create nacos_client_config
    >>> from v2.nacos import ClientConfigBuilder
    >>> config = (ClientConfigBuilder()
    ...     .server_address("localhost:8848")
    ...     .namespace_id("public")
    ...     .username("nacos")
    ...     .password("nacos")
    ...     .build())
    >>> model = NacosChatModel(
    ...     agent_name="my_agent",
    ...     nacos_client_config=config,
    ... )
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

__version__ = "1.0.0"
__author__ = "AgentScope Extension Team"

# =============================================================================
# Core Components - Nacos Service Manager
# =============================================================================
from agentscope_extension_nacos.utils.nacos_service_manager import (
    NacosServiceManager,
    # Convenience functions
    get_nacos_naming_service,
    get_nacos_config_service,
    get_nacos_ai_service,
)

# =============================================================================
# Core Components - Model
# =============================================================================
from agentscope_extension_nacos.model.nacos_chat_model import NacosChatModel

# =============================================================================
# Core Components - Prompt
# =============================================================================
from agentscope_extension_nacos.prompt.nacos_prompt_listener import NacosPromptListener

# =============================================================================
# Core Components - MCP
# =============================================================================
from agentscope_extension_nacos.mcp.agentscope_dynamic_toolkit import DynamicToolkit
from agentscope_extension_nacos.mcp.agentscope_nacos_mcp import (
    NacosHttpStatelessClient,
    NacosHttpStatefulClient,
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
    # Model
    "NacosChatModel",
    # Prompt
    "NacosPromptListener",
    # MCP
    "DynamicToolkit",
    "NacosHttpStatelessClient",
    "NacosHttpStatefulClient",
]
