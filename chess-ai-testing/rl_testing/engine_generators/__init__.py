from typing import Union

from rl_testing.config_parsers import EngineConfig, RemoteEngineConfig
from rl_testing.engine_generators.generators import (
    EngineGenerator,
    RemoteEngineGenerator,
)


def get_engine_generator(
    config: Union[EngineConfig, RemoteEngineConfig]
) -> Union[EngineGenerator, RemoteEngineGenerator]:
    if type(config) == EngineConfig:
        return EngineGenerator(config)
    elif type(config) == RemoteEngineConfig:
        return RemoteEngineGenerator(config)
    else:
        raise ValueError(f"Engine config of type {type(config)} is not supported!")
