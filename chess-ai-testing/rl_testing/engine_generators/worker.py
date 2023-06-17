import argparse
import asyncio
import logging
from multiprocessing import current_process
from pathlib import Path
from typing import Any, Dict, Optional

import chess
import chess.engine

from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.engine_generators.distributed_queue_manager import (
    connect_to_manager,
)
from rl_testing.util.chess import cp2q
from rl_testing.util.util import get_task_result_handler


class AnalysisObject:
    """An object which contains the information required to analyze a position
    and store the results of the analysis.
    """

    def __init__(self, fen: str):
        self.fen = fen
        self.score: Optional[float] = None
        self.best_move: Optional[chess.Move] = None


class TransformationAnalysisObject(AnalysisObject):
    """An analysis object used for the board transformation experiment. Besides the standard
    "fen", "score", and "best_move" attributes, this object also contains the "base_fen" and
    "transformation_index" attributes which are used to match the (potentially transformed)
    board with its original base board and to indicate the transformation that was applied to
    the base board.
    """

    def __init__(self, fen: str, base_fen: str, transformation_index: int):
        self.fen = fen
        self.base_fen = base_fen
        self.transformation_index = transformation_index
        self.score: Optional[float] = None
        self.best_move: Optional[chess.Move] = None


class RecommendedMoveAnalysisObject(AnalysisObject):
    """An analysis object used for the recommended move experiment. Besides the standard
    "fen" and "score" attributes, this object also contains the "parent_fen", "parent_score",
    and "parent_best_move" attributes. All of the parent attributes are used to store the
    results of the analysis of the parent board. The parent board is the original board
    where we apply the recommended move to in order to obtain the child board.
    """

    def __init__(
        self,
        fen: str,
    ):
        self.fen = fen
        self.parent_fen: Optional[str] = None
        self.best_move: Optional[chess.Move] = None
        self.parent_best_move: Optional[chess.Move] = None
        self.score: Optional[float] = None
        self.parent_score: Optional[float] = None

    def prepare_second_round(self) -> None:
        """After the parent board has been analyzed, this method is used to copy all
        stored attributes into the parent attributes in order to make space for the
        results of the analysis of the child board. This method should be called after
        the parent board has been analyzed.
        """
        # Copy the attributes to the parent
        self.parent_fen = self.fen
        self.parent_best_move = self.best_move
        self.parent_score = self.score

        # Get the fen of the board after applying the parent's best move
        board = chess.Board(self.parent_fen)
        board.push(self.parent_best_move)
        new_fen = board.fen(en_passant="fen")

        # Update the current attributes
        self.fen = new_fen
        self.best_move = None
        self.score = None

    def is_result_valid(self) -> bool:
        """Check if the analysis results are valid. The results are valid if neither the
        score nor the best move are None, or "invalid".

        Returns:
            bool: True if the results are valid, False otherwise.
        """
        return isinstance(self.score, float) and isinstance(self.best_move, chess.Move)

    def is_complete(self) -> bool:
        """Check if the analyses of both the parent- and the child board are complete.
        The analysis is complete if all of the attributes have been filled.

        Returns:
            bool: True if the analysis is complete, False otherwise.
        """
        attributes = [
            "fen",
            "parent_fen",
            "best_move",
            "parent_best_move",
            "score",
            "parent_score",
        ]
        return all([getattr(self, attr) is not None for attr in attributes])


async def analyze_position(
    search_limits: Dict[str, Any],
    engine_generator: EngineGenerator,
    engine_config_name: str,
    network_name: Optional[str] = None,
    identifier_str: str = "",
) -> None:
    """A worker task which repeatedly fetches an analysis object from the queue, analyzes
    the position stored in the analysis object, and stores the results of the analysis in
    the analysis object. The analysis object is then added to the result queue.

    Args:
        search_limits (Dict[str, Any]): A dictionary containing the search limits for the
            search engine.
        engine_generator (EngineGenerator): An engine generator which is used to generate
            the search engine.
        engine_config_name (str): The name of the engine config which is used to generate
            the search engine.
        network_name (Optional[str], optional): The name of the config file of the neural
            network weight file which is used by the search engine. Defaults to None.
        identifier_str (str, optional): A string which is used to identify the worker in
            the logs. Defaults to "".
    """
    consumer_queue, producer_queue, required_engine_config = connect_to_manager()

    # Make sure that the worker runs the correct engine config
    if required_engine_config is not None:
        assert (
            engine_config_name == required_engine_config
        ), f"Engine config name mismatch: {engine_config_name} != {required_engine_config}"

    board_counter = 1
    # Required to ensure that the engine doesn't use cached results from
    # previous analyses
    analysis_counter = 0

    # Initialize the engine
    if network_name is not None:
        engine_generator.set_network(network_name=network_name)
    engine = await engine_generator.get_initialized_engine()

    while True:
        # Fetch the next analysis object
        analysis_object: AnalysisObject = consumer_queue.get()
        current_board = chess.Board(analysis_object.fen)

        logging.info(
            f"[{identifier_str}] Analyzing board {board_counter}: "
            + current_board.fen(en_passant="fen")
        )
        try:
            # Analyze the board
            analysis_counter += 1
            info = await engine.analyse(
                current_board, chess.engine.Limit(**search_limits), game=analysis_counter
            )

        except chess.engine.EngineTerminatedError:
            if engine_generator is None:
                logging.info("Can't restart engine due to missing generator")
                raise

            # Try to kill the failed engine
            logging.info(f"[{identifier_str}] Trying to kill engine")
            engine_generator.kill_engine(engine=engine)

            # Try to restart the engine
            logging.info("Trying to restart engine")

            if network_name is not None:
                engine_generator.set_network(network_name=network_name)
            engine = await engine_generator.get_initialized_engine()

            # Add an error to the receiver queue
            analysis_object.score = "invalid"
            analysis_object.best_move = "invalid"
        else:
            # Get the score of the board

            # Get the score of the most promising child board
            score_cp = info["score"].relative.score(mate_score=12780)
            score_q = cp2q(score_cp)
            analysis_object.score = score_q

            # Get the best move
            analysis_object.best_move = info["pv"][0]

        finally:
            # Add the board to the receiver queue
            producer_queue.put(analysis_object)
            consumer_queue.task_done()
            board_counter += 1


async def main(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """The main function used to spawn the worker task.

    Args:
        args (argparse.Namespace): The command line arguments used to configure the worker.
        logger (logging.Logger): The logger used to log messages.
    """
    # Create result directory
    config_folder_path = Path(__file__).absolute().parent.parent.parent / Path(
        "experiments/configs/engine_configs/"
    )

    # Build the engine generator
    engine_config = get_engine_config(
        config_name=args.engine_config_name, config_folder_path=config_folder_path
    )
    assert (
        "verbosemovestats" in engine_config.engine_config
        and engine_config.engine_config["verbosemovestats"]
    ), "VerboseMoveStats must be enabled in the engine config"
    engine_generator = get_engine_generator(engine_config)

    # Get the current process name
    process_name = current_process().name

    # Start the tasks
    analysis_task = asyncio.create_task(
        analyze_position(
            search_limits=engine_config.search_limits,
            engine_generator=engine_generator,
            engine_config_name=args.engine_config_name,
            network_name=args.network_name,
            identifier_str=f"ANALYSIS {process_name}",
        )
    )
    # Add callbacks to the analysis task
    handle_task_exception = get_task_result_handler(
        logger=logger, message="Task raised an exception"
    )
    analysis_task.add_done_callback(handle_task_exception)

    # Wait for data generator task to finish
    await asyncio.wait([analysis_task])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # fmt: off
    parser.add_argument("--seed",                  type=int, default=42)  # noqa
    parser.add_argument("--engine_config_name",    type=str, default="local_400_nodes.ini")  # noqa
    parser.add_argument("--network_name",          type=str,  default="T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207")  # noqa
    # fmt: on
    ##################################
    #           CONFIG END           #
    ##################################
    # Set up the logger
    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()

    # Parse command line arguments
    args = parser.parse_args()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        main(
            args=args,
            logger=logger,
        )
    )
