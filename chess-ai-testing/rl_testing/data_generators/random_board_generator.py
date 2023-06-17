from itertools import count, product
from typing import Optional, Union

import chess
import numpy as np

from rl_testing.config_parsers.data_generator_config_parser import (
    RandomBoardGeneratorConfig,
)
from rl_testing.data_generators.generators import BoardGenerator
from rl_testing.util.chess import is_really_valid

FILES = "abcdefgh"
RANKS = list(range(1, 9))

CHESS_FIELDS = list(product(FILES, RANKS))

CHESS_PIECES_NON_ESSENTIAL = "RNBQBNRPPPPPPPPrnbqbnrpppppppp"

CASTLING_WHITE_KING_SIDE_REQUIRED = [(("e", 1), "K"), (("h", 1), "R")]
CASTLING_WHITE_KING_SIDE_FORBIDDEN = [("f", 1), ("g", 1)]

CASTLING_WHITE_QUEEN_SIDE_REQUIRED = [(("e", 1), "K"), (("a", 1), "R")]
CASTLING_WHITE_QUEEN_SIDE_FORBIDDEN = [("b", 1), ("c", 1), ("d", 1)]

CASTLING_BLACK_KING_SIDE_REQUIRED = [(("e", 8), "k"), (("h", 8), "r")]
CASTLING_BLACK_KING_SIDE_FORBIDDEN = [("f", 8), ("g", 8)]

CASTLING_BLACK_QUEEN_SIDE_REQUIRED = [(("e", 8), "k"), (("a", 8), "r")]
CASTLING_BLACK_QUEEN_SIDE_FORBIDDEN = [("b", 8), ("c", 8), ("d", 8)]


def random_board_position_fen(num_pieces: int, _rng: Optional[np.random.Generator] = None) -> str:
    if _rng is None:
        _rng = np.random.default_rng()

    # Choose the 'num_pieces' chess pieces
    chess_pieces = "Kk" + "".join(
        _rng.choice(list(CHESS_PIECES_NON_ESSENTIAL), num_pieces, replace=False)
    )

    # Choose the position of the pieces
    chess_positions_idx = _rng.choice(
        list(range(len(CHESS_FIELDS))), num_pieces + 2, replace=False
    )
    chess_positions = [CHESS_FIELDS[i] for i in chess_positions_idx]

    # Map the positions to numerical values for convenience
    position_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

    # Create a raw board which will be processed into the real board
    board_raw = [
        (
            position_map[chess_positions[i][0]],
            8 - chess_positions[i][1],
            chess_pieces[i],
        )
        for i in range(len(chess_pieces))
    ]

    # Create a list of positions and pieces
    board_list = list(zip(chess_positions, chess_pieces))

    # Prepare a auxiliary datastructure to convert the board to fen
    position_array = [[] for i in range(8)]
    for file, rank, piece in board_raw:
        position_array[rank].append((file, piece))

    for i in range(8):
        position_array[i] = sorted(position_array[i])

    # Convert the auxiliatry datastructure to fen
    fen_position = ""
    for rank in range(8):
        old_rank = -1
        for rank, piece in position_array[rank]:
            if rank - 1 > old_rank:
                fen_position += str(rank - 1 - old_rank)
            old_rank = rank
            fen_position += piece

        if 8 - 1 > old_rank:
            fen_position += str(8 - 1 - old_rank)
        fen_position += "/"

    fen_position = fen_position[:-1]

    castling_right = ""
    if all([piece in board_list for piece in CASTLING_WHITE_KING_SIDE_REQUIRED]):
        castling_right += "K"
    if all([piece in board_list for piece in CASTLING_WHITE_QUEEN_SIDE_REQUIRED]):
        castling_right += "Q"
    if all([piece in board_list for piece in CASTLING_BLACK_KING_SIDE_REQUIRED]):
        castling_right += "k"
    if all([piece in board_list for piece in CASTLING_BLACK_QUEEN_SIDE_REQUIRED]):
        castling_right += "q"

    if castling_right != "":
        indices = _rng.choice(
            list(range(len(castling_right))), _rng.integers(1, len(castling_right) + 1)
        )
        castling_right = "".join(castling_right[i] for i in indices)

    else:
        castling_right = "-"

    color_to_move = _rng.choice(["w", "b"])

    # Find possible en-passant candidates
    en_passant_candidates = list(
        filter(
            lambda pos: (pos[0][0], pos[1], color_to_move) in [("c", "P", "b"), ("f", "p", "w")],
            board_list,
        )
    )

    if en_passant_candidates:
        en_passant_moves = [e[0][0] + str(e[0][1]) for e in en_passant_candidates]
        # Compute the probability that the last move was a pawn move leading to an
        # en-passant opportunity
        if _rng.random() < 0.5:
            en_passant = _rng.choice(en_passant_moves)
        else:
            en_passant = "-"
    else:
        en_passant = "-"

    half_move = _rng.integers(0, 76)
    full_move = _rng.integers(0, 200)

    fen = (
        fen_position
        + " "
        + color_to_move
        + " "
        + castling_right
        + " "
        + en_passant
        + " "
        + str(half_move)
        + " "
        + str(full_move)
    )

    return fen


def random_valid_board(
    num_pieces: int,
    max_attempts_per_position: Union[int, str] = "unlimited",
    _rng: Optional[np.random.Generator] = None,
) -> Union[chess.Board, str]:
    if _rng is None:
        _rng = np.random.default_rng()
    if max_attempts_per_position == "unlimited":
        attempts = count()
    else:
        attempts = range(max_attempts_per_position)
    for _ in attempts:
        fen = random_board_position_fen(num_pieces=num_pieces, _rng=_rng)
        try:
            board = chess.Board(fen=fen)
            if is_really_valid(board) and board.outcome(claim_draw=True) is None:
                return board
        except ValueError as ve:
            print(ve)

    return "failed"


class RandomBoardGenerator(BoardGenerator):
    def __init__(self, config: RandomBoardGeneratorConfig):
        if config.seed is None:
            self._random_generator = np.random.default_rng()
        else:
            self._random_generator = np.random.default_rng(config.seed)
        self.num_pieces = config.num_pieces
        self.num_pieces_min = config.num_pieces_min
        self.num_pieces_max = config.num_pieces_max
        self.max_attempts_per_position = config.max_attempts_per_position
        self.raise_error_when_failed = config.raise_error_when_failed

    def next(self) -> chess.Board:
        # Choose how many pieces the position should have
        if self.num_pieces is None:
            if self.num_pieces_min is None:
                pieces_min = 1
            else:
                pieces_min = self.num_pieces_min
            if self.num_pieces_max is None:
                pieces_max = len(CHESS_PIECES_NON_ESSENTIAL)
            else:
                pieces_max = self.num_pieces_max
            num_pieces_to_choose = self._random_generator.integers(pieces_min, pieces_max + 1)
        else:
            num_pieces_to_choose = self.num_pieces

        # Create a random chess position
        board_candidate = random_valid_board(
            num_pieces=num_pieces_to_choose,
            max_attempts_per_position=self.max_attempts_per_position,
            _rng=self._random_generator,
        )

        if board_candidate == "failed" and self.raise_error_when_failed:
            raise ValueError("board position could not be generated!")

        return board_candidate


if __name__ == "__main__":
    d = RandomBoardGenerator(
        num_pieces_min=10,
    )

    for board in d:
        print(board.fen())
