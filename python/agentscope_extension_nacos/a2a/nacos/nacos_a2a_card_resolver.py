import asyncio
import logging
from typing import Any, Optional

from a2a.types import AgentCard
from v2.nacos import ClientConfig
from v2.nacos.ai.model.ai_param import GetAgentCardParam, SubscribeAgentCardParam

from agentscope_extension_nacos.a2a.a2a_agent import A2ACardResolverBase
from agentscope_extension_nacos.nacos_service_manager import NacosServiceManager

# Initialize logger
logger = logging.getLogger(__name__)


class NacosA2ACardResolver(A2ACardResolverBase):
	"""Nacos-based A2A Agent Card resolver.

	Resolves and subscribes to agent cards stored in Nacos service registry.
	Supports automatic updates when agent cards change in Nacos.
	Uses NacosServiceManager for connection pooling.

	Args:
		remote_agent_name: Name of the remote agent in Nacos
		nacos_client_config: Optional Nacos client config (uses global if None)
		version: Optional version constraint for the agent card

	Raises:
		ValueError: If remote_agent_name is empty
	"""

	def __init__(
			self,
			remote_agent_name: str,
			nacos_client_config: Optional[ClientConfig] = None,
			version: Optional[str] = None,
	) -> None:
		if not remote_agent_name:
			raise ValueError("remote_agent_name is required")

		self._nacos_client_config: Optional[ClientConfig] = nacos_client_config
		self._remote_agent_name: str = remote_agent_name
		self._version: Optional[str] = version

		# Lazy initialization state
		self._initialized = False
		self._initializing = False
		self._init_lock = asyncio.Lock()

		self._agent_card: AgentCard | None = None

		logger.debug(
			f"[{self.__class__.__name__}] Initialized for agent: {remote_agent_name}")

	async def get_agent_card(
			self,
			relative_card_path: str | None = None,
			http_kwargs: dict[str, Any] | None = None,
	) -> AgentCard:
		"""Get agent card from Nacos with lazy initialization.

		Returns:
			AgentCard: Resolved agent card from Nacos
		"""
		await self._ensure_initialized()
		return self._agent_card

	async def initialize(self):
		"""Public method to trigger explicit initialization."""
		await self._ensure_initialized()

	async def _ensure_initialized(self):
		"""Ensure the resolver is initialized (thread-safe lazy initialization).

		Uses double-checked locking pattern to avoid multiple initializations.
		"""
		if self._initialized:
			return

		# Wait if initialization is in progress
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
				f"[{self.__class__.__name__}] Initializing for agent: {self._remote_agent_name}")
			await self._async_init()
			self._initialized = True
			logger.info(
				f"[{self.__class__.__name__}] Initialization completed for agent: {self._remote_agent_name}")
		except Exception as e:
			logger.error(
				f"[{self.__class__.__name__}] Failed to initialize: {e}")
			raise
		finally:
			self._initializing = False

	async def _async_init(self):
		"""Internal async initialization logic.

		Resolves agent card from Nacos and subscribes to updates.
		Uses NacosServiceManager for connection pooling.
		"""
		# Get Nacos AI service with connection pooling
		manager = NacosServiceManager()
		self._nacos_ai_service = await manager.get_ai_service(
				self._nacos_client_config
		)

		# Fetch agent card from Nacos
		self._agent_card = await self._nacos_ai_service.get_agent_card(
				GetAgentCardParam(
						agent_name=self._remote_agent_name,
						version=self._version,
				))
		logger.info(
			f"[{self.__class__.__name__}] Agent card fetched from Nacos: {self._agent_card.name}")

		# Subscribe to agent card updates
		async def agent_card_subscriber(agent_name: str, agent_card: AgentCard):
			"""Callback for agent card updates from Nacos."""
			logger.info(
				f"[{self.__class__.__name__}] Agent card updated for {agent_name}: {agent_card.name}")
			self._agent_card = agent_card

		await self._nacos_ai_service.subscribe_agent_card(
				SubscribeAgentCardParam(
						agent_name=self._remote_agent_name,
						version=self._version,
						subscribe_callback=agent_card_subscriber
				))
		logger.debug(
			f"[{self.__class__.__name__}] Subscribed to agent card updates for: {self._remote_agent_name}")


