# AgentScope Extensions Nacos

English | [简体中文](./README_CN.md)

An extension component for the [AgentScope](https://github.com/modelscope/agentscope) framework that provides Nacos integration capabilities, supporting dynamic configuration management and MCP tool integration.

## ✨ Key Features

- 🔄 **Dynamic Configuration Management**: Host agent configurations (prompts, model configs, tool lists, etc.) in Nacos for centralized management and real-time hot updates without restarting the application
- 🛠️ **MCP Tool Integration**: Automatically discover and register tool servers from the Nacos MCP Registry with dynamic tool list updates
- 🎯 **Multi-Model Support**: Support for OpenAI, Anthropic, Ollama, Google Gemini, Alibaba Cloud Qwen, and more

## 📋 Prerequisites

- Python >= 3.9
- [AgentScope](https://github.com/modelscope/agentscope) >= 1.0.7
- [AgentScope Runtime](https://github.com/modelscope/agentscope) >= 1.0.1
- Nacos Server >= 3.1.0
- [Nacos Python SDK](https://github.com/nacos-group/nacos-sdk-python) >= 3.0.2

## 📝 Version Compatibility

| Extension Version | AgentScope | AgentScope Runtime | Nacos Server |
|-------------------|------------|--------------------|--------------|
| 1.0.0             | >= 1.0.7   | >= 1.0.1           | >= 3.1.0     |

> **Note**: Starting from version 1.0.0, the A2A protocol implementation has been removed from this extension. AgentScope now natively supports the A2A protocol with Nacos as the A2A Registry. Please use the built-in A2A support in AgentScope directly.

## 📦 Installation

```bash
pip install agentscope-extension-nacos
```

Or install from source:

```bash
git clone https://github.com/nacos-group/agentscope-extensions-nacos.git
cd agentscope-extensions-nacos/python
pip install -e .
```

## 🔧 Configuring Nacos Connection

Before using this extension, you need to configure the Nacos connection information.

### Method 1: Environment Variables

```bash
# Nacos server address (required)
export NACOS_SERVER_ADDRESS=localhost:8848

# Nacos namespace (required)
export NACOS_NAMESPACE_ID=public

# Local Nacos authentication (optional)
export NACOS_USERNAME=nacos
export NACOS_PASSWORD=nacos

# Or use Alibaba Cloud MSE authentication (optional)
export NACOS_ACCESS_KEY=your-access-key
export NACOS_SECRET_KEY=your-secret-key
```

### Method 2: Code Configuration

```python
from v2.nacos import ClientConfigBuilder
from agentscope_extension_nacos.utils.nacos_service_manager import NacosServiceManager

# Configure Nacos connection
client_config = (ClientConfigBuilder()
				 .server_address("localhost:8848")
				 .namespace_id("public")
				 .username("nacos")
				 .password("nacos")
				 .build())

# Set as global configuration
NacosServiceManager.set_global_config(client_config)
```

---

## 🚀 Usage Scenarios

### Scenario 1: Model Configuration Hosting

Host model configuration in Nacos to enable dynamic model switching and parameter adjustment.

#### 1. Create Model Configuration in Nacos

Create the following configuration in the Nacos console:

**Group**: `nacos-ai-model`  
**DataId**: `{model_key}.json` (e.g., `my-model.json`)  
**Format**: JSON

```json
{
  "modelName": "qwen-max",
  "modelProvider": "dashscope",
  "apiKey": "sk-your-api-key",
  "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "args": {
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

**Supported Model Providers**:
- `openai` - OpenAI GPT series
- `anthropic` - Anthropic Claude series
- `ollama` - Ollama local models
- `gemini` - Google Gemini
- `dashscope` - Alibaba Cloud Qwen

#### 2. Use in Code

```python
import asyncio
from v2.nacos import ClientConfigBuilder
from agentscope_extension_nacos.utils.nacos_service_manager import NacosServiceManager
from agentscope_extension_nacos.model.nacos_chat_model import NacosChatModel
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory


async def main():
	# 1. Configure Nacos connection
	client_config = (ClientConfigBuilder()
					 .server_address("localhost:8848")
					 .namespace_id("public")
					 .username("nacos")
					 .password("nacos")
					 .build())
	NacosServiceManager.set_global_config(client_config)

	# 2. Create Nacos-managed model
	model = NacosChatModel(
			model_key="my-model",  # Corresponds to DataId: my-model.json in Group: nacos-ai-model
			stream=True
	)

	# 3. Use in agent
	agent = ReActAgent(
			name="MyAgent",
			sys_prompt="You are an AI assistant",
			model=model,
			formatter=OpenAIChatFormatter(),
			memory=InMemoryMemory()
	)

	# 4. Use the agent
	from agentscope.message import Msg
	response = await agent(Msg(
			name="user",
			content="Hello",
			role="user"
	))
	print(response.content)

	# 5. Cleanup resources
	await NacosServiceManager.cleanup()


if __name__ == "__main__":
	asyncio.run(main())
```

#### 3. Dynamic Model Configuration Updates

After modifying the `model.json` configuration in the Nacos console, the agent will automatically switch to the new model without restarting the application.

---

### Scenario 2: Prompt Configuration Hosting

Host prompt templates in Nacos with support for variable rendering and hot updates.

#### 1. Create Prompt Configuration in Nacos

Create the following configuration in the Nacos console:

**Group**: `nacos-ai-prompt`  
**DataId**: `{prompt_key}.json` (e.g., `my-assistant.json`)  
**Format**: JSON

```json
{
  "template": "You are {{role}}, a helpful assistant specialized in {{domain}}. Please respond in {{language}}."
}
```

The template supports `{{variable}}` syntax for variable rendering.

#### 2. Use in Code

```python
import asyncio
import os
from v2.nacos import ClientConfigBuilder
from agentscope_extension_nacos.utils.nacos_service_manager import NacosServiceManager
from agentscope_extension_nacos.prompt.nacos_prompt_listener import NacosPromptListener
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory


async def main():
    # 1. Configure Nacos connection
    client_config = (ClientConfigBuilder()
                     .server_address("localhost:8848")
                     .namespace_id("public")
                     .username("nacos")
                     .password("nacos")
                     .build())
    NacosServiceManager.set_global_config(client_config)

    # 2. Create Nacos prompt listener with template variables
    prompt_listener = NacosPromptListener(
        prompt_key="my-assistant",  # Corresponds to DataId: my-assistant.json
        args={
            "role": "Jarvis",
            "domain": "programming and technology",
            "language": "English",
        },
    )

    # 3. Create agent
    agent = ReActAgent(
        name="Jarvis",
        sys_prompt="",  # Will be set by NacosPromptListener
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.getenv("DASH_SCOPE_API_KEY"),
        ),
        formatter=DashScopeChatFormatter(),
        memory=InMemoryMemory(),
    )

    # 4. Attach agent to prompt listener and initialize
    prompt_listener.attach_agent(agent)
    await prompt_listener.initialize()

    # Now the agent's sys_prompt is:
    # "You are Jarvis, a helpful assistant specialized in programming and technology. Please respond in English."

    # 5. Cleanup when done
    prompt_listener.detach_agent()
    await NacosServiceManager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
```

#### 3. Dynamic Prompt Updates

After modifying the prompt template in the Nacos console, the agent's prompt will be automatically updated:
- Variables will be re-rendered with the provided `args`
- Agent's `sys_prompt` will be updated in real-time
- No application restart needed

---

### Scenario 3: MCP Tool Integration

Discover and use MCP tool servers from the Nacos MCP Registry.

#### 1. Ensure MCP Server is Registered

MCP servers must be registered in the Nacos MCP Registry first. After registration, they can be used directly in code.

#### 2. Use MCP Tools in Code

```python
import asyncio
from v2.nacos import ClientConfigBuilder
from agentscope_extension_nacos.utils.nacos_service_manager import NacosServiceManager
from agentscope_extension_nacos.mcp.agentscope_nacos_mcp import (
	NacosHttpStatelessClient,
	NacosHttpStatefulClient
)
from agentscope_extension_nacos.mcp.agentscope_dynamic_toolkit import DynamicToolkit
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel


async def main():
	# 1. Configure Nacos connection
	client_config = (ClientConfigBuilder()
					 .server_address("localhost:8848")
					 .namespace_id("public")
					 .username("nacos")
					 .password("nacos")
					 .build())
	NacosServiceManager.set_global_config(client_config)

	# 2. Create MCP clients
	# Stateless client (suitable for low-frequency calls)
	stateless_client = NacosHttpStatelessClient("weather-tools")

	# Stateful client (suitable for high-frequency calls)
	stateful_client = NacosHttpStatefulClient("calculator-tools")

	# 3. Create dynamic toolkit
	toolkit = DynamicToolkit()

	# 4. Register MCP clients
	await stateful_client.connect()
	await toolkit.register_mcp_client(stateless_client)
	await toolkit.register_mcp_client(stateful_client)

	# 5. Use toolkit in agent
	agent = ReActAgent(
			name="ToolAgent",
			sys_prompt="You are an AI assistant that can use tools",
			model=OpenAIChatModel(
					model_name="gpt-4",
					api_key="sk-xxx"
			),
			toolkit=toolkit
	)

	# Tools will automatically sync with Nacos configuration changes
	# No manual refresh needed

	# 6. Cleanup resources
	await stateful_client.close()
	await NacosServiceManager.cleanup()


if __name__ == "__main__":
	asyncio.run(main())
```

#### 3. Dynamic Tool Updates

When MCP server tool configurations are updated in Nacos, `DynamicToolkit` will automatically sync the tool list, and the agent can immediately use the new tools.

---

## 📚 More Examples

Check the [`example/`](./example/) directory for more complete examples:

- [`model_example.py`](./example/model_example.py) - Model configuration and dynamic switching
- [`mcp_example.py`](./example/mcp_example.py) - MCP tool integration example
- [`prompt_example.py`](./example/prompt_example.py) - Prompt configuration hosting with variable rendering

## ⚙️ Advanced Configuration

### NacosChatModel Backup Model

Configure a backup model that automatically falls back when the primary model fails:

```python
from agentscope_extension_nacos.model.nacos_chat_model import NacosChatModel
from agentscope.model import OpenAIChatModel

# Create backup model
backup_model = OpenAIChatModel(
    model_name="gpt-3.5-turbo",
    api_key="sk-xxx"
)

# Create Nacos model (with backup)
model = NacosChatModel(
    agent_name="my-agent",
    nacos_client_config=None,
    stream=True,
    backup_model=backup_model  # Use backup model when primary fails
)
```

### Custom Nacos Configuration

Use different Nacos configurations for different components:

```python
from v2.nacos import ClientConfigBuilder

# Create independent configuration for specific components
custom_config = (ClientConfigBuilder()
    .server_address("another-nacos:8848")
    .namespace_id("test")
    .username("nacos")
    .password("nacos")
    .build())

# Use custom configuration
model = NacosChatModel(
    agent_name="my-agent",
    nacos_client_config=custom_config  # Use custom configuration
)
```

### NacosPromptListener with Custom Args

Dynamically render prompt templates with custom variables:

```python
from agentscope_extension_nacos.prompt.nacos_prompt_listener import NacosPromptListener

# Create prompt listener with template variables
prompt_listener = NacosPromptListener(
    prompt_key="customer-service",
    args={
        "company_name": "Acme Corp",
        "support_hours": "9 AM - 5 PM EST",
        "language": "English",
    },
)
```

## ❓ FAQ

<details>
<summary><b>Q: How to verify successful Nacos connection?</b></summary>

Check the log output for messages like:
```
INFO - [NacosServiceManager] Loaded Nacos config from env (basic auth): localhost:8848
INFO - [NacosServiceManager] NacosServiceManager initialized (singleton)
```

Or verify in code:
```python
manager = NacosServiceManager()
assert manager.is_initialized()
```
</details>

<details>
<summary><b>Q: Configuration not updating after changes in Nacos?</b></summary>

1. Check if Nacos configuration Group and DataId are correct
2. Verify JSON configuration format is valid
3. Check logs for error messages
4. Confirm the listener is properly initialized
</details>

<details>
<summary><b>Q: MCP tools not available?</b></summary>

1. Confirm MCP server is registered in Nacos MCP Registry
2. Check if MCP server is running properly
3. Verify network connectivity
4. Check MCP client logs
</details>

<details>
<summary><b>Q: How to switch between different model providers?</b></summary>

Modify `model.json` configuration in Nacos:
```json
{
  "modelProvider": "openai",  // or "anthropic", "ollama", "gemini", "dashscope"
  "modelName": "gpt-4",
  "apiKey": "sk-xxx"
}
```
The configuration will automatically take effect, and the agent will use the new model provider.
</details>

<details>
<summary><b>Q: What are the naming conventions for agent_name?</b></summary>

agent_name is used to identify configuration groups in Nacos, with the following conventions:
- Only letters, numbers, `.`, `:`, `_`, `-` allowed
- Maximum length 128 characters
- Spaces automatically replaced with underscores
- Configuration Group format: `ai-agent-{agent_name}`
</details>

<details>
<summary><b>Q: How does prompt variable rendering work?</b></summary>

NacosPromptListener uses `{{variable}}` syntax:
- Variables in the template are replaced with values from the `args` dictionary
- If a variable is not found in `args`, the original `{{variable}}` text is kept
- Rendering happens on initial load and whenever the Nacos configuration changes
</details>

## 🤝 Community & Support

- **Issue Reporting**: [GitHub Issues](https://github.com/nacos-group/agentscope-extensions-nacos/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nacos-group/agentscope-extensions-nacos/discussions)
- **AgentScope Docs**: https://github.com/modelscope/agentscope
- **Nacos Docs**: https://nacos.io/docs/

## 📄 License

This project is open-sourced under the [Apache License 2.0](./LICENSE).

## 🙏 Acknowledgments

Thanks to the following projects and communities for their support:
- [AgentScope](https://github.com/modelscope/agentscope) - Powerful multi-agent framework
- [Nacos](https://nacos.io/) - Dynamic service discovery and configuration management platform
- [MCP Protocol](https://modelcontextprotocol.io/) - Model Context Protocol

---

**If this project helps you, please give us a ⭐️ Star!**
