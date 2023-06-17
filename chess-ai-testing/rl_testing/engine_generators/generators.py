import asyncio
import logging
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Tuple, TypeVar

import asyncssh
from asyncssh import SSHSubprocessProtocol, SSHSubprocessTransport
from chess.engine import UciProtocol

from rl_testing.config_parsers.engine_config_parser import (
    EngineConfig,
    RemoteEngineConfig,
)
from rl_testing.engine_generators.relaxed_uci_protocol import (
    RelaxedUciProtocol,
    popen_uci_relaxed,
)

TUciProtocol = TypeVar("TUciProtocol", bound="UciProtocol")


class EngineGenerator:
    def __init__(self, config: EngineConfig) -> None:
        self.engine_path = config.engine_path
        self.network_base_path = config.network_base_path
        self.engine_config = config.engine_config
        self.network_path = config.network_path
        self.initialize_network = config.initialize_network
        self.cp_score_max = config.cp_score_max
        self.cp_score_min = config.cp_score_min
        self.transport_channel_map: Dict[SSHSubprocessProtocol, SSHSubprocessTransport] = {}

    async def _create_engine(
        self, **kwargs: Any
    ) -> Tuple[asyncio.SubprocessTransport, UciProtocol]:
        return await popen_uci_relaxed(
            self.engine_path,
        )

    async def get_initialized_engine(self, **kwargs: Any) -> TUciProtocol:
        if self.initialize_network:
            assert self.network_path is not None, (
                "You first need to set a network using the 'set_network' "
                "function before initializing the engine!"
            )

        # Create the engine
        subprocess_transport, engine = await self._create_engine(**kwargs)

        self.transport_channel_map[engine] = subprocess_transport

        # Initialize the engine
        if not engine.initialized:
            await engine.initialize()

        # Configure engine
        config = dict(self.engine_config)
        if self.initialize_network:
            config["WeightsFile"] = self.network_path

        await engine.configure(config)

        return engine

    def set_network_base_path(self, path: str):
        self.network_base_path = str(Path(path))

    def set_network(self, network_name: str) -> None:
        self.network_path = str(Path(self.network_base_path) / Path(network_name))

    async def close(self):
        pass

    def cp_score_valid(self, score: float) -> bool:
        return self.cp_score_min <= score <= self.cp_score_max

    def kill_engine(self, engine: SSHSubprocessProtocol):
        if engine not in self.transport_channel_map:
            raise ValueError("Engine not found in transport channel map!")

        try:
            self.transport_channel_map[engine].kill()
        except OSError:
            logging.warning("SSH channel already closed!")

        del self.transport_channel_map[engine]

    def kill_all_engines(self):
        engines = list(self.transport_channel_map.keys())
        for engine in engines:
            self.kill_engine(engine)


class RemoteEngineGenerator(EngineGenerator):
    GLOBAL_CONNECTION = None
    GLOBAL_SSH_LOCK = None

    def __init__(self, config: RemoteEngineConfig) -> None:
        super().__init__(config)
        self.remote_host = config.remote_host
        self.remote_user = config.remote_user
        self.password_required = config.password_required

    async def _create_engine(self) -> Tuple[asyncio.SubprocessTransport, RelaxedUciProtocol]:
        if RemoteEngineGenerator.GLOBAL_SSH_LOCK is None:
            RemoteEngineGenerator.GLOBAL_SSH_LOCK = asyncio.Lock()

        async with RemoteEngineGenerator.GLOBAL_SSH_LOCK:
            if RemoteEngineGenerator.GLOBAL_CONNECTION is None:
                # Read in the password from the user
                if self.password_required:
                    remote_password = getpass(
                        prompt="Please specify the SSH password for "
                        f"the user {self.remote_user}:\n"
                    )
                # Start connection
                RemoteEngineGenerator.GLOBAL_CONNECTION = await asyncssh.connect(
                    self.remote_host, username=self.remote_user, password=remote_password
                )

                # Delete the password as quickly as possible
                del remote_password

            return await RemoteEngineGenerator.GLOBAL_CONNECTION.create_subprocess(
                RelaxedUciProtocol,
                self.engine_path,
            )

    async def close(self):
        await RemoteEngineGenerator.GLOBAL_CONNECTION.close()
        RemoteEngineGenerator.GLOBAL_CONNECTION = None
        self.remote_password = None
