from rl_testing.config_parsers.abstract import get_attribute_from_config
from rl_testing.config_parsers.data_generator_config_parser import (
    BoardGeneratorConfig,
    DatabaseBoardGeneratorConfig,
    FENDatabaseBoardGeneratorConfig,
    RandomBoardGeneratorConfig,
    RandomEndgameGeneratorConfig,
)
from rl_testing.config_parsers.engine_config_parser import (
    EngineConfig,
    RemoteEngineConfig,
)


def get_engine_config(config_name: str, config_folder_path: str) -> EngineConfig:
    # Create the config parser
    engine_type = get_attribute_from_config("engine_type", config_name, config_folder_path)

    # Get the engine type
    if engine_type is None:
        raise ValueError(
            "The engine config requires a field 'engine_type' in the section 'General'"
        )
    engine_config_class: EngineConfig

    if engine_type == "local_engine":
        engine_config_class = EngineConfig
    elif engine_type == "remote_engine":
        engine_config_class = RemoteEngineConfig
    else:
        raise ValueError(f"Engine type {engine_type} not supported!")

    engine_config_class.set_config_folder_path(config_folder_path)
    return engine_config_class.from_ini_file(config_name)


def get_data_generator_config(config_name: str, config_folder_path: str) -> BoardGeneratorConfig:
    # Create the config parser
    data_generator_type = get_attribute_from_config(
        "data_generator_type", config_name, config_folder_path
    )

    # Get the engine type
    if data_generator_type is None:
        raise ValueError(
            "The data generator config requires a field 'data_generator_type' "
            "in the section 'General'"
        )

    data_generator_config_class: BoardGeneratorConfig
    if data_generator_type == "board_generator":
        data_generator_config_class = BoardGeneratorConfig
    elif data_generator_type == "database_board_generator":
        data_generator_config_class = DatabaseBoardGeneratorConfig
    elif data_generator_type == "fen_database_board_generator":
        data_generator_config_class = FENDatabaseBoardGeneratorConfig
    elif data_generator_type == "random_board_generator":
        data_generator_config_class = RandomBoardGeneratorConfig
    elif data_generator_type == "random_endgame_generator":
        data_generator_config_class = RandomEndgameGeneratorConfig
    else:
        raise ValueError(f"Engine type {data_generator_type} not supported!")

    data_generator_config_class.set_config_folder_path(config_folder_path)
    return data_generator_config_class.from_ini_file(config_name)
