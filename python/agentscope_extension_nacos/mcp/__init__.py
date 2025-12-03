# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) Module

Provides Nacos-based MCP client implementations supporting:
- Stateless and stateful HTTP clients
- StdIO clients
- Dynamic toolkit (auto-sync with Nacos configuration changes)

Client Types:
- NacosHttpStatelessClient: Stateless HTTP client, creates new connection per call
- NacosHttpStatefulClient: Stateful HTTP client, maintains persistent connection
- NacosStdIOStatefulClient: StdIO client, communicates via standard input/output

Dynamic Toolkit:
- DynamicToolkit: Auto-updating toolkit that syncs when Nacos configuration changes

Usage Examples:
    >>> from agentscope_extension_nacos.mcp import (
    ...     NacosHttpStatelessClient,
    ...     DynamicToolkit,
    ... )
    >>> 
    >>> # Create MCP client with custom nacos_client_config
    >>> from v2.nacos import ClientConfigBuilder
    >>> config = ClientConfigBuilder().server_address("localhost:8848").build()
    >>> client = NacosHttpStatelessClient(
    ...     nacos_client_config=config,
    ...     name="my_mcp_server"
    ... )
    >>> await client.initialize()
    >>> 
    >>> # Create dynamic toolkit
    >>> toolkit = DynamicToolkit()
    >>> await toolkit.register_mcp_client(client)
"""

from agentscope_extension_nacos.mcp.agentscope_nacos_mcp import (
    # Base class
    NacosMCPClientBase,
    # Stateless clients
    NacosHttpStatelessClient,
    # Stateful clients
    NacosStatefulClientBase,
    NacosHttpStatefulClient,
    NacosStdIOStatefulClient,
)

from agentscope_extension_nacos.mcp.agentscope_dynamic_toolkit import (
    DynamicToolkit,
)

__all__ = [
    # Base class
    "NacosMCPClientBase",
    # Stateless clients
    "NacosHttpStatelessClient",
    # Stateful clients
    "NacosStatefulClientBase",
    "NacosHttpStatefulClient",
    "NacosStdIOStatefulClient",
    # Dynamic toolkit
    "DynamicToolkit",
]
