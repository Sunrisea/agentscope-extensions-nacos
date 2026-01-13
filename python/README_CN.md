# AgentScope Extensions Nacos

[English](./README.md) | 简体中文

为 [AgentScope](https://github.com/modelscope/agentscope) 框架提供 Nacos 集成能力的扩展组件，支持动态配置管理和 MCP 工具集成。

## ✨ 核心特性

- 🔄 **动态配置管理**：支持将 Agent 所需的配置（提示词、模型配置、工具列表等）托管至 Nacos，实现集中管理和实时热更新，无需重启应用
- 🛠️ **MCP 工具集成**：自动发现和注册 Nacos MCP Registry 中的工具服务器，工具列表动态更新
- 🎯 **多模型支持**：支持 OpenAI、Anthropic、Ollama、Google Gemini、阿里云通义千问等多种模型

## 📋 前置要求

- Python >= 3.9
- [AgentScope](https://github.com/modelscope/agentscope) >= 1.0.7
- [AgentScope Runtime](https://github.com/modelscope/agentscope) >= 1.0.1
- Nacos Server >= 3.1.0
- [Nacos Python SDK](https://github.com/nacos-group/nacos-sdk-python) >= 3.0.2

## 📝 版本兼容性

| 扩展版本 | AgentScope | AgentScope Runtime | Nacos Server |
|---------|------------|--------------------|--------------|
| 1.0.0   | >= 1.0.7   | >= 1.0.1           | >= 3.1.0     |

> **说明**：从 1.0.0 版本开始，本扩展移除了 A2A 协议的实现。AgentScope 现已原生支持 A2A 协议，并支持使用 Nacos 作为 A2A Registry。请直接使用 AgentScope 内置的 A2A 功能。

## 📦 安装

```bash
pip install agentscope-extension-nacos
```

或从源码安装：

```bash
git clone https://github.com/nacos-group/agentscope-extensions-nacos.git
cd agentscope-extensions-nacos/python
pip install -e .
```

## 🔧 配置 Nacos 连接

在使用本扩展前，首先需要配置 Nacos 连接信息。

### 方式一：环境变量配置

```bash
# Nacos 服务器地址（必需）
export NACOS_SERVER_ADDRESS=localhost:8848

# Nacos 命名空间（必需）
export NACOS_NAMESPACE_ID=public

# 本地 Nacos 认证（可选）
export NACOS_USERNAME=nacos
export NACOS_PASSWORD=nacos

# 或使用阿里云 MSE 认证（可选）
export NACOS_ACCESS_KEY=your-access-key
export NACOS_SECRET_KEY=your-secret-key
```

### 方式二：代码配置

```python
from v2.nacos import ClientConfigBuilder
from agentscope_extension_nacos.utils.nacos_service_manager import NacosServiceManager

# 配置 Nacos 连接
client_config = (ClientConfigBuilder()
				 .server_address("localhost:8848")
				 .namespace_id("public")
				 .username("nacos")
				 .password("nacos")
				 .build())

# 设置为全局配置
NacosServiceManager.set_global_config(client_config)
```

---

## 🚀 使用场景

### 场景一：模型配置托管

将模型配置托管至 Nacos，实现模型的动态切换和参数调整。

#### 1. 在 Nacos 中创建模型配置

在 Nacos 控制台创建以下配置：

**Group**: `nacos-ai-model`  
**DataId**: `{model_key}.json`（例如：`my-model.json`）  
**配置格式**: JSON

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

**支持的模型提供商**：
- `openai` - OpenAI GPT 系列
- `anthropic` - Anthropic Claude 系列
- `ollama` - Ollama 本地模型
- `gemini` - Google Gemini
- `dashscope` - 阿里云通义千问

#### 2. 在代码中使用

```python
import asyncio
from v2.nacos import ClientConfigBuilder
from agentscope_extension_nacos.utils.nacos_service_manager import NacosServiceManager
from agentscope_extension_nacos.model.nacos_chat_model import NacosChatModel
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory


async def main():
	# 1. 配置 Nacos 连接
	client_config = (ClientConfigBuilder()
					 .server_address("localhost:8848")
					 .namespace_id("public")
					 .username("nacos")
					 .password("nacos")
					 .build())
	NacosServiceManager.set_global_config(client_config)

	# 2. 创建 Nacos 管理的模型
	model = NacosChatModel(
			model_key="my-model",  # 对应 DataId: my-model.json
			stream=True
	)

	# 3. 在智能体中使用
	agent = ReActAgent(
			name="MyAgent",
			sys_prompt="你是一个AI助手",
			model=model,
			formatter=OpenAIChatFormatter(),
			memory=InMemoryMemory()
	)

	# 4. 使用智能体
	from agentscope.message import Msg
	response = await agent(Msg(
			name="user",
			content="你好",
			role="user"
	))
	print(response.content)

	# 5. 清理资源
	await NacosServiceManager.cleanup()


if __name__ == "__main__":
	asyncio.run(main())
```

#### 3. 动态更新模型配置

在 Nacos 控制台修改 `model.json` 配置后，智能体会自动切换到新的模型，无需重启应用。

---

### 场景二：Prompt 配置托管

将提示词模板托管至 Nacos，支持变量渲染和热更新。

#### 1. 在 Nacos 中创建 Prompt 配置

在 Nacos 控制台创建以下配置：

**Group**: `nacos-ai-prompt`  
**DataId**: `{prompt_key}.json`（例如：`my-assistant.json`）  
**配置格式**: JSON

```json
{
  "template": "你是{{role}}，一个专注于{{domain}}的智能助手。请使用{{language}}回复。"
}
```

模板支持 `{{变量名}}` 语法进行变量渲染。

#### 2. 在代码中使用

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
    # 1. 配置 Nacos 连接
    client_config = (ClientConfigBuilder()
                     .server_address("localhost:8848")
                     .namespace_id("public")
                     .username("nacos")
                     .password("nacos")
                     .build())
    NacosServiceManager.set_global_config(client_config)

    # 2. 创建 Nacos Prompt 监听器，配置模板变量
    prompt_listener = NacosPromptListener(
        prompt_key="my-assistant",  # 对应 DataId: my-assistant.json
        args={
            "role": "Jarvis",
            "domain": "编程和技术",
            "language": "中文",
        },
    )

    # 3. 创建智能体
    agent = ReActAgent(
        name="Jarvis",
        sys_prompt="",  # 将由 NacosPromptListener 设置
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.getenv("DASH_SCOPE_API_KEY"),
        ),
        formatter=DashScopeChatFormatter(),
        memory=InMemoryMemory(),
    )

    # 4. 将智能体附加到 Prompt 监听器并初始化
    prompt_listener.attach_agent(agent)
    await prompt_listener.initialize()

    # 此时智能体的 sys_prompt 为：
    # "你是Jarvis，一个专注于编程和技术的智能助手。请使用中文回复。"

    # 5. 完成后清理
    prompt_listener.detach_agent()
    await NacosServiceManager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
```

#### 3. 动态更新 Prompt

在 Nacos 控制台修改 Prompt 模板后，智能体的提示词会自动更新：
- 变量会使用提供的 `args` 重新渲染
- 智能体的 `sys_prompt` 会实时更新
- 无需重启应用

---

### 场景三：MCP 工具集成

从 Nacos MCP Registry 中发现和使用 MCP 工具服务器。

#### 1. 确保 MCP 服务器已注册

MCP 服务器需要先在 Nacos MCP Registry 中注册。注册后，可以在代码中直接使用。

#### 2. 在代码中使用 MCP 工具

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
	# 1. 配置 Nacos 连接
	client_config = (ClientConfigBuilder()
					 .server_address("localhost:8848")
					 .namespace_id("public")
					 .username("nacos")
					 .password("nacos")
					 .build())
	NacosServiceManager.set_global_config(client_config)

	# 2. 创建 MCP 客户端
	# 无状态客户端（适合低频调用）
	stateless_client = NacosHttpStatelessClient("weather-tools")

	# 有状态客户端（适合高频调用）
	stateful_client = NacosHttpStatefulClient("calculator-tools")

	# 3. 创建动态工具包
	toolkit = DynamicToolkit()

	# 4. 注册 MCP 客户端
	await stateful_client.connect()
	await toolkit.register_mcp_client(stateless_client)
	await toolkit.register_mcp_client(stateful_client)

	# 5. 在智能体中使用工具包
	agent = ReActAgent(
			name="ToolAgent",
			sys_prompt="你是一个可以使用工具的AI助手",
			model=OpenAIChatModel(
					model_name="gpt-4",
					api_key="sk-xxx"
			),
			toolkit=toolkit
	)

	# 工具会自动同步 Nacos 的配置变更
	# 无需手动刷新

	# 6. 清理资源
	await stateful_client.close()
	await NacosServiceManager.cleanup()


if __name__ == "__main__":
	asyncio.run(main())
```

#### 3. 动态工具更新

当 MCP 服务器的工具配置在 Nacos 中更新时，`DynamicToolkit` 会自动同步工具列表，智能体可以立即使用新的工具。

---

## 📚 更多示例

查看 [`example/`](./example/) 目录获取更多完整示例：

- [`model_example.py`](./example/model_example.py) - 模型配置和动态切换
- [`mcp_example.py`](./example/mcp_example.py) - MCP 工具集成示例
- [`prompt_example.py`](./example/prompt_example.py) - Prompt 配置托管与变量渲染

## ⚙️ 高级配置

### NacosChatModel 备用模型

配置备用模型，当主模型失败时自动降级：

```python
from agentscope_extension_nacos.model.nacos_chat_model import NacosChatModel
from agentscope.model import OpenAIChatModel

# 创建备用模型
backup_model = OpenAIChatModel(
    model_name="gpt-3.5-turbo",
    api_key="sk-xxx"
)

# 创建 Nacos 模型（带备用）
model = NacosChatModel(
    agent_name="my-agent",
    nacos_client_config=None,
    stream=True,
    backup_model=backup_model  # 主模型失败时使用备用模型
)
```

### 自定义 Nacos 配置

为不同组件使用不同的 Nacos 配置：

```python
from v2.nacos import ClientConfigBuilder

# 为特定组件创建独立配置
custom_config = (ClientConfigBuilder()
    .server_address("another-nacos:8848")
    .namespace_id("test")
    .username("nacos")
    .password("nacos")
    .build())

# 使用自定义配置
model = NacosChatModel(
    agent_name="my-agent",
    nacos_client_config=custom_config  # 使用自定义配置
)
```

### NacosPromptListener 自定义变量

使用自定义变量动态渲染 Prompt 模板：

```python
from agentscope_extension_nacos.prompt.nacos_prompt_listener import NacosPromptListener

# 创建 Prompt 监听器，配置模板变量
prompt_listener = NacosPromptListener(
    prompt_key="customer-service",
    args={
        "company_name": "阿里云",
        "support_hours": "9:00 - 18:00",
        "language": "中文",
    },
)
```

## ❓ 常见问题

<details>
<summary><b>Q: 如何验证 Nacos 连接是否成功？</b></summary>

检查日志输出，应该看到类似以下信息：
```
INFO - [NacosServiceManager] Loaded Nacos config from env (basic auth): localhost:8848
INFO - [NacosServiceManager] NacosServiceManager initialized (singleton)
```

或者在代码中验证：
```python
manager = NacosServiceManager()
assert manager.is_initialized()
```
</details>

<details>
<summary><b>Q: 配置更新后没有生效？</b></summary>

1. 检查 Nacos 配置的 Group 和 DataId 是否正确
2. 检查配置 JSON 格式是否正确
3. 查看日志是否有错误信息
4. 确认监听器已正确初始化
</details>

<details>
<summary><b>Q: MCP 工具不可用？</b></summary>

1. 确认 MCP 服务器已在 Nacos MCP Registry 中注册
2. 检查 MCP 服务器是否正常运行
3. 验证网络连接是否正常
4. 查看 MCP 客户端日志
</details>

<details>
<summary><b>Q: 如何切换不同的模型提供商？</b></summary>

在 Nacos 中修改 `model.json` 配置：
```json
{
  "modelProvider": "openai",  // 或 "anthropic", "ollama", "gemini", "dashscope"
  "modelName": "gpt-4",
  "apiKey": "sk-xxx"
}
```
配置会自动生效，智能体会使用新的模型提供商。
</details>

<details>
<summary><b>Q: agent_name 有什么命名规范？</b></summary>

agent_name 用于在 Nacos 中标识配置组，命名规范：
- 只能包含字母、数字、`.`、`:`、`_`、`-`
- 最大长度 128 字符
- 空格会自动替换为下划线
- 配置 Group 格式为：`ai-agent-{agent_name}`
</details>

<details>
<summary><b>Q: Prompt 变量渲染是如何工作的？</b></summary>

NacosPromptListener 使用 `{{变量名}}` 语法：
- 模板中的变量会被 `args` 字典中的值替换
- 如果变量在 `args` 中不存在，会保留原始的 `{{变量名}}` 文本
- 渲染发生在初始加载时以及 Nacos 配置变更时
</details>

## 🤝 社区与支持

- **问题反馈**：[GitHub Issues](https://github.com/nacos-group/agentscope-extensions-nacos/issues)
- **讨论交流**：[GitHub Discussions](https://github.com/nacos-group/agentscope-extensions-nacos/discussions)
- **AgentScope 文档**：https://github.com/modelscope/agentscope
- **Nacos 文档**：https://nacos.io/docs/

## 📄 许可证

本项目基于 [Apache License 2.0](./LICENSE) 开源。

## 🙏 致谢

感谢以下项目和社区的支持：
- [AgentScope](https://github.com/modelscope/agentscope) - 强大的多智能体框架
- [Nacos](https://nacos.io/) - 动态服务发现和配置管理平台
- [MCP Protocol](https://modelcontextprotocol.io/) - 模型上下文协议

---

**如果这个项目对您有帮助，请给我们一个 ⭐️ Star！**
