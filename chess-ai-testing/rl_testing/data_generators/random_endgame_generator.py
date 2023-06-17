from itertools import count, product
from typing import Optional, Union

import chess
import numpy as np
from rl_testing.config_parsers import RandomEndgameGeneratorConfig
from rl_testing.data_generators import BoardGenerator
from rl_testing.util.chess import is_really_valid, has_undefended_attacked_pieces

CHESS_PIECES_WHITE_NON_ESSENTIAL = "RNBQBNRPPPPPPPP"
CHESS_PIECES_BLACK_NON_ESSENTIAL = "rnbqbnrpppppppp"
CHESS_PIECES_NON_ESSENTIAL = CHESS_PIECES_WHITE_NON_ESSENTIAL + CHESS_PIECES_BLACK_NON_ESSENTIAL
CHESS_PIECES_WHITE_NON_ESSENTIAL_NO_PAWNS = "RNBQBNR"
CHESS_PIECES_BLACK_NON_ESSENTIAL_NO_PAWNS = "rnbqbnr"
CHESS_PIECES_NON_ESSENTIAL_NO_PAWNS = (
    CHESS_PIECES_WHITE_NON_ESSENTIAL_NO_PAWNS + CHESS_PIECES_BLACK_NON_ESSENTIAL_NO_PAWNS
)


def random_endgame_fen_candidate(
    num_pieces: int,
    no_pawns: bool,
    color_balance: bool,
    identical_pieces: bool,
    _rng: Optional[np.random.Generator] = None,
) -> str:
    if _rng is None:
        _rng = np.random.default_rng()

    if color_balance and num_pieces % 2 != 0:
        num_pieces += 1

    # The two kings are always present
    piece_selection = "Kk"
    num_pieces -= 2

    pieces_white = ""
    pieces_black = ""

    # Create the pieces string from which we will randomly select pieces
    if no_pawns:
        pieces_white += CHESS_PIECES_WHITE_NON_ESSENTIAL_NO_PAWNS
        pieces_black += CHESS_PIECES_BLACK_NON_ESSENTIAL_NO_PAWNS
    else:
        pieces_white += CHESS_PIECES_WHITE_NON_ESSENTIAL
        pieces_black += CHESS_PIECES_BLACK_NON_ESSENTIAL

    # Choose the pieces
    if color_balance or identical_pieces:
        piece_selection += "".join(
            _rng.choice(list(pieces_white), size=num_pieces // 2, replace=False)
        )
        if identical_pieces:
            piece_selection += piece_selection[2:].lower()
        else:
            piece_selection += "".join(
                _rng.choice(list(pieces_black), size=num_pieces // 2, replace=False)
            )
    else:
        piece_selection += "".join(
            _rng.choice(list(pieces_white + pieces_black), size=num_pieces, replace=False)
        )

    # Convert the piece selection to a list of chess.Piece objects
    piece_selection = [chess.Piece.from_symbol(piece) for piece in piece_selection]

    # Choose the positions
    positions = _rng.choice(chess.SQUARES, size=num_pieces + 2, replace=False)

    # Create the piece-map
    piece_map = dict(zip(positions, piece_selection))

    # Create the FEN
    board = chess.Board()
    board.set_piece_map(piece_map)
    board.fullmove_number = _rng.integers(30, 71)
    board.halfmove_clock = _rng.integers(0, board.fullmove_number // 3)
    board.set_castling_fen("")

    return board.fen()


def random_valid_board(
    num_pieces: int,
    no_pawns: bool,
    no_free_pieces: bool,
    color_balance: bool,
    identical_pieces: bool,
    max_attempts_per_position: int,
    _rng: Optional[np.random.Generator] = None,
) -> Union[chess.Board, str]:
    if _rng is None:
        _rng = np.random.default_rng()
    if max_attempts_per_position == "unlimited":
        attempts = count()
    else:
        attempts = range(max_attempts_per_position)
    for _ in attempts:
        fen = random_endgame_fen_candidate(
            num_pieces=num_pieces,
            no_pawns=no_pawns,
            color_balance=color_balance,
            identical_pieces=identical_pieces,
            _rng=_rng,
        )
        try:
            board = chess.Board(fen=fen)
            if is_really_valid(board) and board.outcome(claim_draw=True) is None:
                if no_free_pieces:
                    if not has_undefended_attacked_pieces(board):
                        return board
                else:
                    return board
        except ValueError as ve:
            print(ve)

    return "failed"


class RandomEndgameGenerator(BoardGenerator):
    def __init__(self, config: RandomEndgameGeneratorConfig):
        if config.seed is None:
            self._random_generator = np.random.default_rng()
        else:
            self._random_generator = np.random.default_rng(config.seed)
        self.num_pieces = config.num_pieces
        self.num_pieces_min = config.num_pieces_min
        self.num_pieces_max = config.num_pieces_max
        self.no_pawns = config.no_pawns
        self.color_balance = config.color_balance
        self.identical_pieces = config.identical_pieces
        self.no_free_pieces = config.no_free_pieces
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
                pieces_max = len(
                    CHESS_PIECES_NON_ESSENTIAL_NO_PAWNS
                    if self.no_pawns
                    else CHESS_PIECES_NON_ESSENTIAL
                )
            else:
                pieces_max = self.num_pieces_max
            num_pieces_to_choose = self._random_generator.integers(pieces_min, pieces_max + 1)
        else:
            num_pieces_to_choose = self.num_pieces

        # Create a random chess position
        board_candidate = random_valid_board(
            num_pieces=num_pieces_to_choose,
            no_pawns=self.no_pawns,
            no_free_pieces=self.no_free_pieces,
            color_balance=self.color_balance,
            identical_pieces=self.identical_pieces,
            max_attempts_per_position=self.max_attempts_per_position,
            _rng=self._random_generator,
        )

        if board_candidate == "failed" and self.raise_error_when_failed:
            raise ValueError("board position could not be generated!")

        return board_candidate
