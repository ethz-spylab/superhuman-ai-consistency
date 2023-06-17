from rl_testing.config_parsers import (
    BoardGeneratorConfig,
    DatabaseBoardGeneratorConfig,
    FENDatabaseBoardGeneratorConfig,
    RandomBoardGeneratorConfig,
    RandomEndgameGeneratorConfig,
)
from rl_testing.data_generators.database_position_generator import (
    DatabaseBoardGenerator,
)
from rl_testing.data_generators.fen_database_position_generator import (
    FENDatabaseBoardGenerator,
)
from rl_testing.data_generators.generators import BoardGenerator
from rl_testing.data_generators.random_board_generator import RandomBoardGenerator
from rl_testing.data_generators.random_endgame_generator import RandomEndgameGenerator


def get_data_generator(config: BoardGeneratorConfig) -> BoardGenerator:
    if type(config) == DatabaseBoardGeneratorConfig:
        return DatabaseBoardGenerator(config)
    elif type(config) == FENDatabaseBoardGeneratorConfig:
        return FENDatabaseBoardGenerator(config)
    elif type(config) == RandomBoardGeneratorConfig:
        return RandomBoardGenerator(config)
    elif type(config) == RandomEndgameGeneratorConfig:
        return RandomEndgameGenerator(config)
    else:
        raise ValueError(f"Engine config of type {type(config)} is not supported!")
