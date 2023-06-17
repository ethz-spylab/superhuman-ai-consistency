from pathlib import Path

from rl_testing.config_parsers import get_data_generator_config
from rl_testing.data_generators import get_data_generator

if __name__ == "__main__":
    DATA_CONFIG_NAME = "random_many_pieces.ini"
    NUM_POSITIONS_TO_CREATE = 100000
    FILE_NAME = "data/random_positions.txt"

    data_config = get_data_generator_config(
        DATA_CONFIG_NAME,
        Path(__file__).parent.parent.absolute()
        / Path("experiments/configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    with open(FILE_NAME, "a") as f:
        boards_read = 0
        boards_found: set = set()
        for i in range(NUM_POSITIONS_TO_CREATE):
            while True:
                if boards_read % 10000 == 0:
                    print(f"Scanned {boards_read} boards")
                board = data_generator.next()
                boards_read += 1
                fen = board.fen(en_passant="fen")
                if fen not in boards_found:
                    boards_found.add(fen)
                    break

            print(
                f"Created random position {i+1}/{NUM_POSITIONS_TO_CREATE}: {fen} "
                f"after scanning {boards_read} boards."
            )

            f.write(f"{fen}\n")
