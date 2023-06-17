import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import chess.svg
import pandas as pd
from chess import flip_anti_diagonal, flip_diagonal, flip_horizontal, flip_vertical
from chess.engine import Score

from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.mcts.tree_parser import NodeInfo, TreeInfo
from rl_testing.util.chess import (
    apply_transformation,
    cp2q,
    plot_two_boards,
    rotate_90_clockwise,
    rotate_180_clockwise,
    rotate_270_clockwise,
)

transformation_dict = {
    "rot90": rotate_90_clockwise,
    "rot180": rotate_180_clockwise,
    "rot270": rotate_270_clockwise,
    "flip_diag": flip_diagonal,
    "flip_anti_diag": flip_anti_diagonal,
    "flip_hor": flip_horizontal,
    "flip_vert": flip_vertical,
    "mirror": "mirror",
}


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, help="Path to result file which stores the results that should be analyzed", required=True)  # noqa
    parser.add_argument("--score_type1", type=str, help="Whether the score of the root node should be extracted or the score of the best move for each first position", required=False, choices=["node", "best_move"], default="best_move")  # noqa
    parser.add_argument("--score_type2", type=str, help="Whether the score of the root node should be extracted or the score of the best move for each second position", required=False, choices=["node", "best_move"], default="node")  # noqa
    parser.add_argument("--build_fens_from_transformations", action="store_true", help="Whether the fens should be built from transformations", required=False, default=False)  # noqa
    parser.add_argument("--num_examples", type=int, help="Number of examples to plot", required=False, default=10)  # noqa
    parser.add_argument("--engine_config_name", type=str, help="Name of the engine config to use", required=True)  # noqa
    parser.add_argument("--network_name", type=str, default="T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207")  # noqa
    parser.add_argument("--fen_column1", type=str, help="Name of the column storing the first fen value", required=False, default="parent_fen")  # noqa
    parser.add_argument("--fen_column2", type=str, help="Name of the column storing the second fen value", required=False, default="child_fen")  # noqa
    parser.add_argument("--score_column1", type=str, help="Name of the column storing the first score value. Only required if the fens are built from transformations.", required=False, default=None)  # noqa
    parser.add_argument("--flip_second_score", action="store_true", help="Whether the second Q-value should be flipped (multiplied by -1)", required=False, default=False)  # noqa
    parser.add_argument("--show_best_move_first", action="store_true", help="Whether the best move should be shown for the first position", required=False, default=False)  # noqa
    parser.add_argument("--show_best_move_second", action="store_true", help="Whether the best move should be shown for the second position", required=False, default=False)  # noqa
    parser.add_argument("--large_fontsize", action="store_true", help="Whether the fontsize should be increased", required=False, default=False)  # noqa
    parser.add_argument("--save_plot", action="store_true", help="Save the resulting plots to a file", required=False, default=False)  # noqa
    parser.add_argument("--save_path_base", type=str, help="Base path to use for saving the plots", required=False, default=None)  # noqa
    parser.add_argument("--show_plot", action="store_true", help="Show the resulting plots", required=False, default=False)  # noqa
    # fmt: on
    return parser.parse_args()


def build_fens_from_transformations(
    dataframe: pd.DataFrame,
    first_fen_column: str,
    first_score_column: str,
    original_fens: List[str],
) -> Tuple[List[str], List[str]]:
    # Only extract the lines of the dataframe that contain one of the first fens
    dataframe = dataframe[dataframe[first_fen_column].isin(original_fens)].copy()

    # Get all column names which are also keys in the transformation dictionary
    transformation_names = [
        column_name for column_name in dataframe.columns if column_name in transformation_dict
    ]

    if len(transformation_names) == 0:
        raise ValueError(
            "No transformation names found! Make sure that the column names of the dataframe"
            " are also keys in the transformation dictionary"
        )

    # Get all columns which store scores
    score_names = [first_score_column] + list(transformation_names)
    score_columns = dataframe[score_names]

    # For each row, store the column name storing the lowest score and the column name storing
    # the highest score
    dataframe["min_score_column"] = score_columns.idxmin(axis=1)
    dataframe["max_score_column"] = score_columns.idxmax(axis=1)

    # Iterate over all rows and build the first- and second fens
    first_fens = []
    second_fens = []
    for index, row in dataframe.iterrows():
        # Get the transformation name
        row_transformation_names = [
            row["min_score_column"],
            row["max_score_column"],
        ]

        # Build the two fens from the two transformations
        fens = []
        for transformation_name in row_transformation_names:
            # Get the original fen of this row
            original_fen = row[first_fen_column]
            # If the transformation name is just first_score_column, then the fen is the original fen
            if transformation_name == first_score_column:
                fens.append(row[first_fen_column])
            else:
                # First build the original board which will then be transformed
                original_board = chess.Board(fen=original_fen)

                # Get the transformation function
                transformed_board = apply_transformation(
                    original_board, transformation_dict[transformation_name]
                )

                # Add the fen to the list
                fens.append(transformed_board.fen(en_passant="fen"))

        # Add the fens to the list
        first_fens.append(fens[0])
        second_fens.append(fens[1])

    return first_fens, second_fens


def build_second_fens(original_fens: List[str], transformation_name: str) -> List[str]:
    assert transformation_name.startswith(
        "create_"
    ), "Invalid transformation name! Must start with 'create_'"

    # Remove the prefix
    transformation_name = transformation_name.replace("create_", "")

    assert transformation_name in transformation_dict, (
        f"Invalid transformation name: {transformation_name}. Must be one of"
        f" {list(transformation_dict.keys())}"
    )

    # Get the transformation function
    transformation_function = transformation_dict[transformation_name]

    # Build the second fens
    second_fens = []
    for fen in original_fens:
        board = chess.Board(fen=fen)
        second_board = apply_transformation(board, transformation_function)
        second_fens.append(second_board.fen(en_passant="fen"))

    return second_fens


async def analyze_with_engine(
    engine_generator: EngineGenerator,
    positions: List[Union[chess.Board, str]],
    network_name: Optional[str] = None,
    search_limits: Optional[Dict[str, Any]] = None,
    score_type: str = "cp",
) -> Tuple[List[Score], List[TreeInfo]]:
    valid_score_types = ["cp", "q"]
    engine_scores = []
    verbose_stats = []

    # Set search limits
    if search_limits is None:
        search_limits = {"depth": 25}

    # Setup and configure the engine
    if network_name is not None:
        engine_generator.set_network(network_name)
    engine = await engine_generator.get_initialized_engine()

    for board_index, board in enumerate(positions):
        # Make sure that 'board' has type 'chess.Board'
        if isinstance(board, str):
            fen = board
            board = chess.Board(fen=fen)
        else:
            fen = board.fen(en_passant="fen")

        # Analyze the position
        print(f"Analyzing board {board_index+1}/{len(positions)}: {fen}")
        info = await engine.analyse(board, chess.engine.Limit(**search_limits))

        # Extract the score
        cp_score = info["score"].relative.score(mate_score=12780)
        if score_type == "cp":
            engine_scores.append(cp_score)
        elif score_type == "q":
            engine_scores.append(cp2q(cp_score))
        else:
            raise ValueError(
                f"Invalid score type: {score_type}. Choose one from {valid_score_types}"
            )

        if "root_and_child_scores" in info:
            verbose_stats.append(info["root_and_child_scores"])

    return engine_scores, verbose_stats


def analyze_positions(
    fens: List[str],
    args: argparse.Namespace,
) -> List[NodeInfo]:
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    # Setup engine generator
    engine_config = get_engine_config(
        config_name=args.engine_config_name,
        config_folder_path=Path(__file__).parent.parent.absolute()
        / Path("configs/engine_configs/"),
    )
    search_limit = engine_config.search_limits
    engine_generator = get_engine_generator(engine_config)

    _, verbose_stats = asyncio.run(
        analyze_with_engine(
            engine_generator=engine_generator,
            positions=fens,
            network_name=args.network_name,
            search_limits=search_limit,
        )
    )

    return verbose_stats


def extract_win_prob_and_best_move_from_verbose_stat(
    verbose_stat: NodeInfo, mode: str, flip_q_value: bool = False
) -> Tuple[float, chess.Move]:
    valid_modes = ["node", "best_move"]
    assert mode in valid_modes, f"Invalid mode: {mode}. Choose one from {valid_modes}"

    # Get the best move
    best_edge = max(verbose_stat.child_edges, key=lambda s: s.num_visits + s.in_flight_visits)
    best_move = best_edge.move

    # Extract the win probability
    if mode == "node":
        q_value = verbose_stat.q_value
        d_value = verbose_stat.draw_value
    elif mode == "best_move":
        q_value = best_edge.q_value
        d_value = best_edge.draw_value

    print(f"fen: {verbose_stat.fen}, q_value: {q_value}, d_value: {d_value}")

    if flip_q_value:
        q_value *= -1

    win_prob = 0.5 * (q_value + 1 - d_value)

    return win_prob, best_move


def create_two_board_plot(
    first_fen: str,
    second_fen: str,
    first_win_prob: float,
    second_win_prob: float,
    second_win_prob_flipped: float,
    best_move_first: Optional[chess.Move] = None,
    best_move_second: Optional[chess.Move] = None,
    large_font_size: bool = False,
    show_plot: bool = True,
    save_plot: bool = False,
    save_path: str = "",
):
    if save_plot:
        assert save_path, "If save_plot is True, save_path must be specified!"

    # Create the boards
    first_board = chess.Board(first_fen)
    second_board = chess.Board(second_fen)

    # Extract the colors of the players to move
    first_player_to_move = "White" if first_board.turn == chess.WHITE else "Black"
    second_player_to_move = "White" if second_board.turn == chess.WHITE else "Black"
    first_color = first_player_to_move
    if second_win_prob_flipped:
        second_color = "White" if second_player_to_move == "Black" else "Black"
    else:
        second_color = second_player_to_move

    first_win_prob = str(round(100 * first_win_prob))
    second_win_prob = str(round(100 * second_win_prob))

    # Compute the number of digits in the win probabilities
    first_len = len(str(first_win_prob))
    second_len = len(str(second_win_prob))

    # Create the titles
    if large_font_size:
        first_len += 1 if first_len == 2 else 0
        second_len += 1 if second_len == 2 else 0

        title1 = f"{first_player_to_move} to move"
        title2 = f"{second_player_to_move} to move"

        # Compute the correct amount of spaces to add
        win_prob_string = "Win prob:"
        first_spaces = " " * (7 - first_len)
        second_spaces = " " * (7 - second_len)

    else:
        title1 = f"Board 1: {first_player_to_move} to move"
        title2 = f"Board 2: {second_player_to_move} to move"

        # Compute the correct amount of spaces to add
        win_prob_string = "Win probability:"
        first_spaces = " " * (13 - first_len)
        second_spaces = " " * (13 - second_len)

    # Create the x-axis labels
    x_labels = []
    for win_prob, space, color, best_move, board in zip(
        [first_win_prob, second_win_prob],
        [first_spaces, second_spaces],
        [first_color, second_color],
        [best_move_first, best_move_second],
        [first_board, second_board],
    ):
        x_label = f"{win_prob_string}{space}{win_prob}% for {color}"
        if best_move:
            first_line_length = len(x_label)
            move_san = board.san(best_move)
            san_length = len(move_san)
            second_line_space = 21 if san_length == 3 else 20
            second_line = f"\nBest move:{' ' * (second_line_space - san_length)}{move_san}"
            second_line_length = len(second_line)
            x_label += second_line
            if first_line_length > second_line_length:
                x_label += " " * (first_line_length - second_line_length)
        x_labels.append(x_label)

    # Create the x-axis labels
    x_label1, x_label2 = x_labels

    # Build the arrows of the best moves
    arrows1 = []
    arrows2 = []
    if best_move_first:
        arrows1 = [
            chess.svg.Arrow(best_move_first.from_square, best_move_first.to_square, color="green")
        ]
    if best_move_second:
        arrows2 = [
            chess.svg.Arrow(best_move_second.from_square, best_move_second.to_square, color="green")
        ]

    if large_font_size:
        fontsize = 18
    else:
        fontsize = 14

    # Create the plots
    plot_two_boards(
        board1=first_board,
        board2=second_board,
        arrows1=arrows1,
        arrows2=arrows2,
        title1=title1,
        title2=title2,
        x_label1=x_label1,
        x_label2=x_label2,
        fontsize=fontsize,
        plot_size=800,
        save=save_plot,
        show=show_plot,
        save_path=save_path,
    )


def plot_interesting_examples(args: argparse.Namespace):
    # Load the data
    dataframe = pd.read_csv(args.result_path)

    # Make sure that the dataframe is sorted by the difference column
    dataframe = dataframe.sort_values(by="difference", ascending=False)

    # Extract the FENs
    first_fens = dataframe[[args.fen_column1]].values.transpose().tolist()[0]
    if args.build_fens_from_transformations:
        first_fens, second_fens = build_fens_from_transformations(
            dataframe=dataframe,
            first_fen_column=args.fen_column1,
            first_score_column=args.score_column1,
            original_fens=first_fens[: args.num_examples],
        )
    else:
        second_fens = dataframe[[args.fen_column2]].values.transpose().tolist()[0]

    # Extract only the first args.num_examples
    first_fens = first_fens[: args.num_examples]
    second_fens = second_fens[: args.num_examples]

    # Analyze the positions
    all_fens = first_fens + second_fens
    all_stats = analyze_positions(all_fens, args)
    first_stats, second_stats = (
        all_stats[: args.num_examples],
        all_stats[args.num_examples :],
    )

    # Extract the win probabilities
    print("First Scores:")
    first_win_probs_and_best_moves = []
    for index, stat in enumerate(first_stats):
        stat.set_fen(first_fens[index])
        first_win_probs_and_best_moves.append(
            extract_win_prob_and_best_move_from_verbose_stat(stat, args.score_type1)
        )

    print("Second Scores:")
    second_win_probs_and_best_moves = []
    for index, stat in enumerate(second_stats):
        stat.set_fen(second_fens[index])
        second_win_probs_and_best_moves.append(
            extract_win_prob_and_best_move_from_verbose_stat(
                stat, args.score_type2, args.flip_second_score
            )
        )
    first_win_probs, first_best_moves = zip(*first_win_probs_and_best_moves)
    second_win_probs, second_best_moves = zip(*second_win_probs_and_best_moves)

    # Create the plots
    for index, (
        first_fen,
        second_fen,
        first_win_prob,
        second_win_prob,
        first_best_move,
        second_best_move,
    ) in enumerate(
        zip(
            first_fens,
            second_fens,
            first_win_probs,
            second_win_probs,
            first_best_moves,
            second_best_moves,
        )
    ):
        if args.large_fontsize:
            save_path = str(args.save_path_base) + f"_{index+1}_large.png"
        else:
            save_path = str(args.save_path_base) + f"_{index+1}.png"
        print(f"Plotting example {index+1}/{args.num_examples}")
        create_two_board_plot(
            first_fen=first_fen,
            second_fen=second_fen,
            first_win_prob=first_win_prob,
            second_win_prob=second_win_prob,
            second_win_prob_flipped=args.flip_second_score,
            best_move_first=first_best_move if args.show_best_move_first else None,
            best_move_second=second_best_move if args.show_best_move_second else None,
            large_font_size=args.large_fontsize,
            show_plot=args.show_plot,
            save_plot=args.save_plot,
            save_path=save_path,
        )


if __name__ == "__main__":
    # Get all command-line arguments
    args = parse_args()

    # Run the analysis
    plot_interesting_examples(args)
