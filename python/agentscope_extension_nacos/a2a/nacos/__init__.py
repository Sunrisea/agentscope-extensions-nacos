# -*- coding: utf-8 -*-
"""
A2A Nacos Integration Module

Provides deep integration between A2A protocol and Nacos:
- Automatic Agent Card registration to Nacos
- Agent Card resolution from Nacos
- Service endpoint auto-registration and discovery

Main Classes:
- A2AFastAPINacosAdaptor: FastAPI adapter that exposes Agent as A2A service
                          and registers to Nacos
- NacosA2ACardResolver: Agent Card resolver that fetches from Nacos

Usage Examples:
    >>> from agentscope_extension_nacos.a2a.nacos import (
    ...     A2AFastAPINacosAdaptor,
    ...     NacosA2ACardResolver,
    ... )
    >>> 
    >>> # Server side: Expose Agent as A2A service with custom nacos_client_config
    >>> from v2.nacos import ClientConfigBuilder
    >>> config = ClientConfigBuilder().server_address("localhost:8848").build()
    >>> adaptor = A2AFastAPINacosAdaptor(
    ...     agent=my_agent,
    ...     nacos_client_config=config,  # Pass custom config
    ...     host="0.0.0.0",
    ...     port=8090,
    ... )
    >>> 
    >>> # Client side: Get Agent Card from Nacos
    >>> resolver = NacosA2ACardResolver(
    ...     remote_agent_name="remote_agent",
    ...     nacos_client_config=config,  # Pass custom config
    ... )
    >>> from agentscope_extension_nacos.a2a import A2aAgent
    >>> agent = A2aAgent(agent_card_source=None, agent_card_resolver=resolver)
"""

from agentscope_extension_nacos.a2a.nacos.nacos_a2a_adapter import (
    A2AFastAPINacosAdaptor,
)

from agentscope_extension_nacos.a2a.nacos.nacos_a2a_card_resolver import (
    NacosA2ACardResolver,
)

__all__ = [
    "A2AFastAPINacosAdaptor",
    "NacosA2ACardResolver",
]
