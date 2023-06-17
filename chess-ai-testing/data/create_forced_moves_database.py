import argparse
from pathlib import Path

from rl_testing.config_parsers import get_data_generator_config
from rl_testing.data_generators import get_data_generator


def has_at_least_k_pieces(board, k):
    num_fields_occupied = 0
    occupied_bitboard = board.occupied
    for _ in range(k):
        if occupied_bitboard == 0:
            break
        occupied_bitboard &= occupied_bitboard - 1
        num_fields_occupied += 1
    else:
        # If we get here, we have at least min_pieces pieces
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add database config file parameter
    parser.add_argument(
        "--data_config",
        type=str,
        default="database_late_moves.ini",
        help="The name of the database config file to use.",
    )
    # Add number of positions to create parameter
    parser.add_argument(
        "--num_positions",
        type=int,
        default=100000,
        help="The number of positions to create.",
    )
    # Add output file name parameter
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/forced_move_positions_leela.txt",
        help="The name of the output file to write the positions to.",
    )
    # Add minimum number of pieces parameter
    parser.add_argument(
        "--min_pieces",
        type=int,
        default=8,
        help="The minimum number of pieces to have on the board.",
    )

    args = parser.parse_args()

    data_config_name = args.data_config
    num_positions_to_create = args.num_positions
    output_file = args.output_file
    min_pieces = args.min_pieces
    assert 2 <= min_pieces <= 32, "min_pieces must be between 1 and 32."

    data_config = get_data_generator_config(
        data_config_name,
        Path(__file__).parent.parent.absolute()
        / Path("experiments/configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    with open(output_file, "a") as f:
        boards_read = 0
        boards_found: set = set()
        for i in range(num_positions_to_create):
            while True:
                if boards_read % 10000 == 0:
                    print(f"Scanned {boards_read} boards")
                board = data_generator.next()
                boards_read += 1

                # Check that the board is a forced move position
                if board.legal_moves.count() == 1:
                    fen = board.fen(en_passant="fen")

                    # Check that the game is not over after the only legal move
                    legal_moves = list(board.legal_moves)
                    assert len(legal_moves) == 1
                    move = legal_moves[0]
                    board.push(move)
                    if board.is_game_over():
                        continue

                    if fen not in boards_found:
                        if has_at_least_k_pieces(board, min_pieces):
                            boards_found.add(fen)
                            break

            print(
                f"Found forced move {i+1}/{num_positions_to_create}: {fen} "
                f"after scanning {boards_read} boards."
            )

            f.write(f"{fen}\n")
