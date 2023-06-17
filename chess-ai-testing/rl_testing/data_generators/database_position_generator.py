import os
from pathlib import Path

import chess
import chess.pgn

from rl_testing.config_parsers.data_generator_config_parser import (
    DatabaseBoardGeneratorConfig,
)
from rl_testing.data_generators.generators import BoardGenerator

if "DATASET_PATH" in os.environ:
    DATA_PATH = Path(os.environ["DATASET_PATH"]) / "chess-data"
else:
    DATA_PATH = data_path = Path(__file__).parent.parent.parent / "data"


class DatabaseBoardGenerator(BoardGenerator):
    def __init__(
        self,
        config: DatabaseBoardGeneratorConfig,
    ):
        self.data_path = DATA_PATH / config.database_name
        self.file_iterator = None
        self.get_positions_after_move = config.get_positions_after_move

        self.current_game = None
        self.current_board = None
        self.games_read = config.games_read
        self.moves_read = 0

        if config.open_now:
            self.setup_position()

    def setup_position(self) -> None:
        self.file_iterator = open(self.data_path, "r")

        # Read all games which have already been read
        for _ in range(self.games_read):
            self.current_game = chess.pgn.read_game(self.file_iterator)

        if self.current_game is None:
            self.current_game = chess.pgn.read_game(self.file_iterator)
            self.games_read += 1

        # Read new games as long as the current game has not enough moves
        while (
            len(list(self.current_game.mainline_moves()))
            < self.get_positions_after_move * 2 + self.moves_read
        ):
            self.current_game = chess.pgn.read_game(self.file_iterator)
            self.games_read += 1

        if "FEN" in self.current_game.headers:
            self.current_board = chess.Board(self.current_game.headers["FEN"])
        else:
            self.current_board = chess.Board()
        moves = list(self.current_game.mainline_moves())

        # Read all moves which have already been read
        for move_index, move in enumerate(moves):
            if move_index >= self.get_positions_after_move * 2 + self.moves_read:
                break
            self.current_board.push(move=move)

    def close_database(self):
        if self.file_iterator is not None:
            self.file_iterator.close()
        self.file_iterator = None

    def next(self) -> chess.Board:
        # If the file has been closed, re-open it
        if self.file_iterator is None:
            self.setup_position()

        moves = list(self.current_game.mainline_moves())

        # If all moves of this game have been read, load a new game
        # Because some games are empty, repeat this process until you
        # find a game which is not empty
        while self.moves_read + 2 * self.get_positions_after_move >= len(moves):
            self.games_read += 1
            self.moves_read = 0
            self.current_game = chess.pgn.read_game(self.file_iterator)
            if "FEN" in self.current_game.headers:
                self.current_board = chess.Board(self.current_game.headers["FEN"])
            else:
                self.current_board = chess.Board()
            moves = list(self.current_game.mainline_moves())

        for move_idx in range(
            len(self.current_board.move_stack), 2 * self.get_positions_after_move
        ):
            self.current_board.push(move=moves[move_idx])

        # Read the next move
        self.current_board.push(moves[self.moves_read + 2 * self.get_positions_after_move])
        self.moves_read += 1

        # Return the new position
        return self.current_board.copy()


if __name__ == "__main__":
    """
    TODO:
        1. Test implementation
        2. Add a parameter which only reads in moves after a certain depth
    """
    gen = DatabaseBoardGenerator(database_name="caissabasse.pgn", get_positions_after_move=15)
    for i in range(50):
        board = gen.next()
        print(board.fen())
