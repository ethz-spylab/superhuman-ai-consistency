import os
from pathlib import Path

import chess

from rl_testing.config_parsers.data_generator_config_parser import (
    FENDatabaseBoardGeneratorConfig,
)
from rl_testing.data_generators.generators import BoardGenerator

if "DATASET_PATH" in os.environ:
    DATA_PATH = Path(os.environ["DATASET_PATH"]) / "chess-data"
else:
    DATA_PATH = data_path = Path(__file__).parent.parent.parent / "data"


class FENDatabaseBoardGenerator(BoardGenerator):
    def __init__(self, config: FENDatabaseBoardGeneratorConfig):
        self.data_path = DATA_PATH / config.database_name
        self.file_iterator = None

        self.positions_read = 0

        if config.open_now:
            self.setup_position()

    def setup_position(self) -> None:
        self.file_iterator = open(self.data_path, "r")

        # Read all games which have already been read
        for _ in range(self.positions_read):
            self.current_board = chess.Board(self.file_iterator.readline())

    def close_database(self):
        if self.file_iterator is not None:
            self.file_iterator.close()
        self.file_iterator = None

    def next(self) -> chess.Board:
        # If the file has been closed, re-open it
        if self.file_iterator is None:
            self.setup_position()

        # Read the next move
        self.current_board = chess.Board(self.file_iterator.readline())
        self.positions_read += 1

        # Return the new position
        return self.current_board


if __name__ == "__main__":
    """
    TODO:
        1. Test implementation
        2. Add a parameter which only reads in moves after a certain depth
    """
    gen = FENDatabaseBoardGenerator(database_name="random_positions.txt")
    for i in range(50):
        board = gen.next()
        print(board.fen())
