from typing import Generator, List, Optional, Union

import chess.pgn


def is_middle_game_position(
    board: chess.Board, middle_game_start: int = 15, min_pieces: int = 0
) -> bool:
    """Check if a position is a middle game position. A position is considered a middle game
    position if it happened

    Args:
        board (chess.Board): Board object.
        middle_game_start (int, optional): Move number at which the middle game starts. Defaults to 15.
        min_pieces (int, optional): Minimum number of pieces required on the board. Defaults to 0.

    Returns:
        bool: Whether the position is a middle game position.
    """
    if len(board.piece_map()) < min_pieces:
        return False
    return board.fullmove_number >= middle_game_start and not is_endgame_position(board)


def is_endgame_position(position: Union[str, chess.Board], min_pieces: int = 0) -> bool:
    """Check if a position is an endgame position. A position is considered an endgame position if
    one of the following conditions is met:
    - There are less than 10 pieces on the board.
    - Both queens have been captured and there are less than or equal to 6 non-pawn pieces on
      the board.
    - There are less than 3 non-pawn and non-king pieces per side on the board.

    Args:
        position (Union[str, chess.Board]): FEN string or chess.Board object.
        min_pieces (int, optional): Minimum number of pieces required on the board. Defaults to 0.

    Returns:
        bool: Whether the position is an endgame position.
    """
    if isinstance(position, str):
        board = chess.Board(position)
    else:
        board = position

    piece_map = board.piece_map()

    if len(piece_map) < min_pieces:
        return False

    pieces = list(piece_map.values())
    num_queens = sum([1 for piece in pieces if piece.piece_type == chess.QUEEN])
    num_special_pieces = sum(
        [
            1
            for piece in pieces
            if piece.piece_type in [chess.BISHOP, chess.KNIGHT, chess.ROOK, chess.QUEEN]
        ]
    )

    if len(piece_map) < 10:
        return True

    elif num_queens == 0 and num_special_pieces <= 6:
        return True

    elif num_special_pieces < 6:
        return True

    return False


def get_fens_from_pgn(
    pgn_path: str,
    mode: str = "all",
    min_pieces: int = 0,
    max_num: Optional[int] = None,
    use_dict: bool = True,
) -> Generator[str, None, None]:
    """Get a list of FENs from a PGN file.

    Args:
        pgn_path (str): Path to the PGN file.
        mode (str, optional): Mode to extract FENs. Must be one of ["all", "middlegame", "endgame"].
            Defaults to "all".
        min_pieces (int, optional): Minimum number of pieces required on the board. Defaults to 0.
        max_num (int, optional): Maximum number of FENs to extract. Defaults to None.
        use_dict (bool, optional): Whether to use a dictionary to store the FENs. Defaults to True.

    Yields:
        Generator[str, None, None]: Generator of FENs.
    """
    fen_cache = set()
    with open(pgn_path) as pgn:
        while max_num is None or len(fen_cache) < max_num:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()

                # Make sure that the game is not over
                if board.is_game_over():
                    break

                if mode == "middlegame" and not is_middle_game_position(
                    board, min_pieces=min_pieces
                ):
                    if is_endgame_position(board, min_pieces=0):
                        break
                    continue

                elif mode == "endgame" and not is_endgame_position(board, min_pieces=min_pieces):
                    continue

                if use_dict and fen not in fen_cache:
                    fen_cache.add(fen)
                    yield fen

                elif not use_dict:
                    yield fen


def convert_pgn_to_fens(
    pgn_path: str,
    output_path: str,
    mode: str = "all",
    min_pieces: int = 0,
    max_num: Optional[int] = None,
):
    """Convert a PGN file to a list of FENs.

    Args:
        pgn_path (str): Path to the PGN file.
        output_path (str): Path to the output file.
        mode (str, optional): Mode to extract FENs. Must be one of ["all", "middlegame", "endgame"].
            Defaults to "all".
        min_pieces (int, optional): Minimum number of pieces required on the board. Defaults to 0.
        max_num (int, optional): Maximum number of FENs to extract. Defaults to None.
    """
    assert mode in [
        "all",
        "middlegame",
        "endgame",
    ], f"Invalid mode {mode}. Must be one of ['all', 'middlegame', 'endgame']"

    with open(output_path, "w") as f:
        for index, fen in enumerate(
            get_fens_from_pgn(pgn_path, mode=mode, min_pieces=min_pieces, max_num=max_num)
        ):
            if index % 1000 == 0:
                print(f"Processed {index} FENs")
            f.write(f"{fen}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn_path", type=str, help="Path to the PGN file.")
    parser.add_argument("--output_path", type=str, help="Path to the output file.")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "middlegame", "endgame"],
        help=(
            "Mode to extract FENs. Must be one of ['all', 'middlegame', 'endgame']. Defaults to"
            " 'all'."
        ),
    )
    parser.add_argument(
        "--min_pieces",
        type=int,
        default=0,
        help="Minimum number of pieces required on the board. Defaults to 0.",
    )
    parser.add_argument(
        "--max_num",
        type=int,
        default=None,
        help="Maximum number of FENs to extract. Defaults to None.",
    )
    args = parser.parse_args()

    convert_pgn_to_fens(
        args.pgn_path,
        args.output_path,
        mode=args.mode,
        min_pieces=args.min_pieces,
        max_num=args.max_num,
    )
