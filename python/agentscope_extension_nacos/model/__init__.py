# -*- coding: utf-8 -*-
"""
Model Module

Provides Nacos-based dynamic chat model configuration supporting:
- Dynamic model configuration loading from Nacos
- Auto-update on configuration changes
- Multiple model providers (OpenAI, Anthropic, Ollama, Gemini, DashScope)
- Automatic formatter selection

Main Classes:
- NacosChatModel: Dynamically configured chat model with Nacos integration
- AutoFormatter: Automatic formatter that selects based on model provider

Usage Examples:
    >>> from agentscope_extension_nacos.model import (
    ...     NacosChatModel,
    ...     AutoFormatter,
    ... )
    >>> 
    >>> # Create dynamic chat model with custom nacos_client_config
    >>> from v2.nacos import ClientConfigBuilder
    >>> config = ClientConfigBuilder().server_address("localhost:8848").build()
    >>> model = NacosChatModel(
    ...     agent_name="my_agent",
    ...     nacos_client_config=config,  # Pass custom config
    ...     stream=True,
    ... )
    >>> await model.initialize()
    >>> 
    >>> # Create auto formatter
    >>> formatter = AutoFormatter(if_multi_agent=False, chat_model=model)
    >>> 
    >>> # Use the model
    >>> response = await model(messages)
"""

from agentscope_extension_nacos.model.nacos_chat_model import (
    NacosChatModel,
    AutoFormatter,
)

__all__ = [
    "NacosChatModel",
    "AutoFormatter",
]
