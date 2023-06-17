import configparser
from pathlib import Path
from typing import Any, Dict, Union

from rl_testing.config_parsers.abstract import Config


class BoardGeneratorConfig(Config):
    CONFIG_FOLDER = Path("./configs/data_generator_configs")
    DEFAULT_CONFIG_NAME = Path("random_default.ini")
    REQUIRED_ATTRIBUTES = ["data_generator_type"]
    OPTIONAL_ATTRIBUTES = []

    def __init__(
        self,
        config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser],
        _initialize: bool = True,
    ):
        # Initialize the parameters
        self.board_generator_config = {}

        # Assign the parameters from the provided config file
        if _initialize:
            self.set_parameters(config=config)
            self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        if (
            name in BoardGeneratorConfig.REQUIRED_ATTRIBUTES
            or name in BoardGeneratorConfig.OPTIONAL_ATTRIBUTES
        ):
            setattr(self, name, self.parse_string(value, raise_error=False))
        else:
            raise ValueError(f"Objects of type {type(self)} don't have the attribute {name}!")


class DatabaseBoardGeneratorConfig(BoardGeneratorConfig):
    REQUIRED_ATTRIBUTES = ["database_name"]
    OPTIONAL_ATTRIBUTES = ["open_now", "get_positions_after_move", "games_read"]

    def __init__(
        self,
        config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser],
    ):
        super().__init__(config=config, _initialize=False)

        self.database_name = None
        self.open_now = True
        self.get_positions_after_move = 0
        self.games_read = 0

        self.set_parameters(config=config)
        self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        if (
            name in DatabaseBoardGeneratorConfig.REQUIRED_ATTRIBUTES
            or name in DatabaseBoardGeneratorConfig.OPTIONAL_ATTRIBUTES
        ):
            setattr(self, name, self.parse_string(value, raise_error=False))
        else:
            super().set_parameter(section, name, value)


class FENDatabaseBoardGeneratorConfig(BoardGeneratorConfig):
    REQUIRED_ATTRIBUTES = ["database_name"]
    OPTIONAL_ATTRIBUTES = ["open_now"]

    def __init__(
        self,
        config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser],
    ):
        super().__init__(config=config, _initialize=False)

        self.database_name = None
        self.open_now = True

        self.set_parameters(config=config)
        self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        if (
            name in FENDatabaseBoardGeneratorConfig.REQUIRED_ATTRIBUTES
            or name in FENDatabaseBoardGeneratorConfig.OPTIONAL_ATTRIBUTES
        ):
            setattr(self, name, self.parse_string(value, raise_error=False))
        else:
            super().set_parameter(section, name, value)


class RandomBoardGeneratorConfig(BoardGeneratorConfig):
    REQUIRED_ATTRIBUTES = []
    OPTIONAL_ATTRIBUTES = [
        "num_pieces",
        "num_pieces_min",
        "num_pieces_max",
        "max_attempts_per_position",
        "raise_error_when_failed",
        "seed",
    ]

    def __init__(
        self,
        config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser],
        _initialize: bool = True,
    ):
        super().__init__(config=config, _initialize=False)

        self.num_pieces = None
        self.num_pieces_min = None
        self.num_pieces_max = None
        self.max_attempts_per_position = 100000
        self.raise_error_when_failed = False
        self.seed = None

        if _initialize:
            self.set_parameters(config=config)
            self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        if (
            name in RandomBoardGeneratorConfig.REQUIRED_ATTRIBUTES
            or name in RandomBoardGeneratorConfig.OPTIONAL_ATTRIBUTES
        ):
            setattr(self, name, self.parse_string(value, raise_error=False))
        else:
            super().set_parameter(section, name, value)


class RandomEndgameGeneratorConfig(RandomBoardGeneratorConfig):
    OPTIONAL_ATTRIBUTES = [
        "no_pawns",
        "no_free_pieces",
        "color_balance",
        "identical_pieces",
    ]

    def __init__(
        self,
        config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser],
    ):
        super().__init__(config=config, _initialize=False)

        self.num_pieces = None
        self.num_pieces_min = None
        self.num_pieces_max = None
        self.no_pawns = False
        self.no_free_pieces = False
        self.color_balance = False
        self.identical_pieces = False
        self.max_attempts_per_position = 100000
        self.raise_error_when_failed = False
        self.seed = None

        self.set_parameters(config=config)
        self.check_parameters()

    def set_parameter(self, section: str, name: str, value: str) -> None:
        if (
            name in RandomEndgameGeneratorConfig.REQUIRED_ATTRIBUTES
            or name in RandomEndgameGeneratorConfig.OPTIONAL_ATTRIBUTES
        ):
            setattr(self, name, self.parse_string(value, raise_error=False))
        else:
            super().set_parameter(section, name, value)


if __name__ == "__main__":
    c = BoardGeneratorConfig.default_config()
    print("finished")
