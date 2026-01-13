import asyncio
import json
import logging
import re
from typing import Optional

from agentscope.agent import ReActAgent
from v2.nacos import ClientConfig, NacosConfigService, ConfigParam

from agentscope_extension_nacos.utils.nacos_service_manager import NacosServiceManager

# Initialize logger
logger = logging.getLogger(__name__)

class NacosPromptListener:

	def __init__(
			self,
			prompt_key: str,
			nacos_client_config: Optional[ClientConfig] = None,
			args : dict | None = None,
	):

		self._initialized = False
		self._initializing = False
		self._init_lock = asyncio.Lock()

		self.prompt_key = prompt_key
		self._nacos_client_config: Optional[ClientConfig] = nacos_client_config
		self.nacos_config_service: NacosConfigService | None = None

		self.args: dict = args or {}
		self.agent : ReActAgent | None = None


	async def _ensure_initialized(self):
		"""Ensure ChatModel is initialized (thread-safe lazy initialization).

		Uses double-checked locking pattern to avoid race conditions.
		"""
		if self._initialized:
			return

		# If initializing, wait for completion
		if self._initializing:
			while self._initializing:
				await asyncio.sleep(0.01)
			return

		async with self._init_lock:
			# Double-check to avoid duplicate initialization
			if self._initialized:
				return

			self._initializing = True
			try:
				logger.info(
					f"[{self.__class__.__name__}] Starting initialization for prompt: {self.prompt_key}")
				await self._async_init()
				self._initialized = True
				logger.info(
					f"[{self.__class__.__name__}] Successfully initialized for prompt: {self.prompt_key}")
			except Exception as e:
				logger.error(
					f"[{self.__class__.__name__}] Initialization failed for prompt {self.prompt_key}: {e}",
					exc_info=True)
				raise
			finally:
				self._initializing = False

	async def _async_init(self):
		"""Internal async initialization logic.

		Loads model configuration from Nacos and sets up configuration listeners.
		"""
		# Use NacosServiceManager to get service (automatically reuses connections)
		manager = NacosServiceManager()
		self.nacos_config_service = await manager.get_config_service(
				self._nacos_client_config)

		user_prompt_config_group_name = "nacos-ai-prompt"
		user_prompt_config_data_id = f"{self.prompt_key}.json"
		user_prompt_config = await self.nacos_config_service.get_config(
				ConfigParam(
						data_id=user_prompt_config_data_id,
						group=user_prompt_config_group_name,
				))

		if user_prompt_config is None or len(user_prompt_config) == 0:
			logger.error(
				f"[{self.__class__.__name__}] No config found for prompt {self.prompt_key}")
			raise Exception(
					f"No config found for prompt {self.prompt_key}")

		prompt_config = json.loads(user_prompt_config)
		self.prompt_template = prompt_config.get("template", "")
		self._set_prompt(self._render_template(self.prompt_template))


		async def promot_listener(tenant, data_id, group, content):
			"""Listener for user model configuration changes"""
			logger.info(
				f"[{self.__class__.__name__}] User model config changed - data_id: {data_id}, group: {group}")
			try:
				if content is None or len(content) == 0:
					logger.error(
						f"[{self.__class__.__name__}] Invalid config for prompt {self.prompt_key}")
					return
				prompt_config = json.loads(content)
				self.prompt_template = prompt_config.get("template", "")
				self._set_prompt(self._render_template(self.prompt_template))


				logger.info(
					f"[{self.__class__.__name__}] Prompt updated successfully")
			except Exception as e:
				logger.error(
					f"[{self.__class__.__name__}] Failed to update prompt from config change: {e}")

				raise Exception(
							f"Failed to create prompt for {self.prompt_key}: {e}")

		await self.nacos_config_service.add_listener(
				data_id=user_prompt_config_data_id,
				group=user_prompt_config_group_name,
				listener=promot_listener)
		logger.debug(
			f"[{self.__class__.__name__}] Registered user model config listener")

	async def initialize(self):
		"""Public initialization method (maintains backward compatibility).

		Callers can explicitly call this method for initialization,
		or skip it (will auto-initialize lazily when needed).
		"""
		await self._ensure_initialized()


	def attach_agent(self, agent: ReActAgent):
		"""Attach agent to the listener."""
		self.agent = agent

	def detach_agent(self):
		"""Detach agent from the listener."""
		self.agent = None

	def _render_template(self, template: str) -> str:
		"""Render template by replacing {{variable}} with values from args."""
		def replace_var(match):
			var_name = match.group(1).strip()
			return str(self.args.get(var_name, match.group(0)))
		
		return re.sub(r'\{\{(.+?)\}\}', replace_var, template)

	def _set_prompt(self, prompt:str):
		if self.agent is not None:
			self.agent._sys_prompt = prompt