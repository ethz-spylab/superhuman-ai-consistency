import configparser
from pathlib import Path
from typing import Any, Dict, Union

from rl_testing.config_parsers.abstract import Config

EngineConfigs = Union["EngineConfig", "RemoteEngineConfig"]

# TODO REMOVE UNNECESSARY FUNCTIONS


class EngineConfig(Config):
    CONFIG_FOLDER = Path("./configs/engine_configs")
    DEFAULT_CONFIG_NAME = Path("default_local.ini")
    REQUIRED_ATTRIBUTES = [
        "engine_type",
        "engine_path",
        "network_base_path",
        "engine_config",
        "search_limits",
    ]
    OPTIONAL_ATTRIBUTES = ["network_path", "initialize_network", "cp_score_max", "cp_score_min"]

    def __init__(
        self,
        config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser],
        _initialize: bool = True,
    ):
        # Initialize the parameters
        self.engine_type = None
        self.engine_path = None
        self.network_base_path = None
        self.network_path = None
        self.engine_config = {}
        self.search_limits = {}
        self.initialize_network = True
        self.cp_score_max = 12801
        self.cp_score_min = -12801

        # Assign the parameters from the provided config file
        if _initialize:
            self.set_parameters(config=config)
            self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        value = self.parse_string(value, raise_error=False)
        if section == "EngineConfig":
            self.engine_config[name] = value
        elif section == "SearchLimits":
            self.search_limits[name] = value
        elif name in EngineConfig.REQUIRED_ATTRIBUTES or name in EngineConfig.OPTIONAL_ATTRIBUTES:
            setattr(self, name, value)
        else:
            raise ValueError(f"Objects of type {type(self)} don't have the attribute {name}!")

    def set_network(self, network_name: str) -> None:
        self.network_path = str(Path(self.network_base_path) / Path(network_name))


class RemoteEngineConfig(EngineConfig):
    DEFAULT_CONFIG_NAME = Path("default_remote.ini")
    REQUIRED_ATTRIBUTES = ["remote_host", "remote_user", "password_required"]
    OPTIONAL_ATTRIBUTES = ["network_path"]

    def __init__(self, config: Union[Dict[str, Any], configparser.ConfigParser]):
        super().__init__(config=config, _initialize=False)

        self.remote_host = None
        self.remote_user = None
        self.password_required = None

        self.set_parameters(config=config)
        self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        value = self.parse_string(value, raise_error=False)
        if (
            name in RemoteEngineConfig.REQUIRED_ATTRIBUTES
            or name in RemoteEngineConfig.OPTIONAL_ATTRIBUTES
        ):
            setattr(self, name, value)
        else:
            super().set_parameter(section, name, value)

    def check_parameters(self) -> None:
        for name in self.REQUIRED_ATTRIBUTES:
            if not hasattr(self, name) or getattr(self, name) is None:
                raise ValueError(f"Parameter '{name}' must be defined!")
        super().check_parameters()


if __name__ == "__main__":
    r = RemoteEngineConfig.default_config()
    s = RemoteEngineConfig.from_ini_file("remote_400_nodes.ini")
    print("finished")
