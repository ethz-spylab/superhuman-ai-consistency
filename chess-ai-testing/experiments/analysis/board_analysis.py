import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import netwulf as nw
from chess.engine import Score

from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.mcts.tree_parser import (
    TreeInfo,
    convert_tree_to_networkx,
    plot_networkx_tree,
)
from rl_testing.util.chess import cp2q


async def analyze_with_engine(
    engine_generator: EngineGenerator,
    positions: List[Union[chess.Board, str]],
    network_name: Optional[str] = None,
    search_limits: Optional[Dict[str, Any]] = None,
    score_type: str = "cp",
) -> Tuple[List[Score], List[TreeInfo]]:
    valid_score_types = ["cp", "q"]
    engine_scores = []
    trees = []

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

        if "mcts_tree" in info:
            trees.append(info["mcts_tree"])

    return engine_scores, trees


if __name__ == "__main__":
    ################
    # CONFIG START #
    ################
    fens = [
        # "1r6/pN4K1/3R4/n1R2qp1/PpPp2qB/6p1/p1kPpp2/bB5r b - - 46 117"
        "r5k1/r5Pp/8/1q2p3/4Q3/1p6/p2R4/K5R1 b - - 2 39",
        "4r1k1/r5Pp/8/1q2p3/4Q3/1p6/p2R4/K5R1 w - - 3 40",
    ]

    network_name = (
        # "T785469-600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2"
        "T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207"
    )
    # network_name = None
    plot_graph = True
    score_type = "q"

    # engine_config_name = "remote_25_depth_stockfish.ini"
    # search_limit = {"depth": 40}
    # engine_config_name = "remote_debug_500_nodes.ini"
    # engine_config_name = "remote_400_nodes.ini"
    engine_config_name = "remote_debug_400_nodes.ini"
    search_limit = {"nodes": 400}
    ################
    #  CONFIG END  #
    ################

    # Analyze the positions with stockfish
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    # Setup engine generator
    engine_config = get_engine_config(
        config_name=engine_config_name,
        config_folder_path=Path(__file__).parent.parent.absolute()
        / Path("configs/engine_configs/"),
    )
    engine_generator = get_engine_generator(engine_config)

    scores, trees = asyncio.run(
        analyze_with_engine(
            engine_generator=engine_generator,
            positions=fens,
            network_name=network_name,
            search_limits=search_limit,
            score_type=score_type,
        )
    )

    print("Results:")
    index = 0
    for fen, score in zip(fens, scores):
        print(f"board {fen :<74} score: {score}")
        if plot_graph:
            tree = trees[index]
            graph = convert_tree_to_networkx(tree, only_basic_info=True)
            print(f"Number of nodes: {len(graph.nodes)}")
            print(f"Number of edges: {len(graph.edges)}")
            # stylized_network, config = nw.visualize(graph)
            plot_networkx_tree(tree, only_basic_info=True)
        index += 1
