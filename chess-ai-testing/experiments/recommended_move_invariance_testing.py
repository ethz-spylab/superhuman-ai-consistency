import argparse
import asyncio
import logging
import os
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Union

import chess
import chess.engine
import numpy as np

from rl_testing.config_parsers import get_data_generator_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.util.experiment import store_experiment_params
from rl_testing.util.util import get_task_result_handler
from rl_testing.engine_generators.distributed_queue_manager import (
    QueueManager,
    address,
    connect_to_manager,
    password,
    port,
)
from rl_testing.engine_generators.worker import (
    RecommendedMoveAnalysisObject,
)

RESULT_DIR = Path(__file__).parent / Path("results/recommended_move_testing")


async def create_positions(
    data_generator: BoardGenerator,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    identifier_str: str = "",
) -> None:
    """Create chess positions using the provided data generator and send the results to the
    output queue.

    Args:
        data_generator (BoardGenerator): A BoardGenerator object that is used to create the
            chess positions.
        num_positions (int, optional): The number of chess positions to create. Defaults to 1.
        sleep_between_positions (float, optional): The number of seconds to wait between creating
            two chess positions. Useful to pause this async function and allow other async
            functions to run. Defaults to 0.1.
        identifier_str (str, optional): A string that is used to identify this process.
            Defaults to "".
    """
    fen_cache: Set[str] = set()

    output_queue, _, _ = connect_to_manager()

    board_index = 1
    while board_index <= num_positions:
        # Create a random chess position
        board_candidate = data_generator.next()

        # Get the number of legal moves for the current board
        if board_candidate == "failed":
            continue

        legal_moves = list(board_candidate.legal_moves)

        # Check if one of the legal moves results in a checkmate for the current player
        # If so, we do not want to use this position
        should_break = False
        for move in legal_moves:
            board_candidate.push(move)
            if board_candidate.is_checkmate():
                should_break = True
                board_candidate.pop()
                break
            board_candidate.pop()

        if should_break:
            continue

        # Check if the generated position should be further processed
        if board_candidate.fen() not in fen_cache and 0 < len(legal_moves):
            fen_cache.add(board_candidate.fen())

            # Log the base position
            fen = board_candidate.fen(en_passant="fen")
            logging.info(f"[{identifier_str}] Created base board {board_index + 1}: {fen}")

            output_queue.put(RecommendedMoveAnalysisObject(fen=fen))

            await asyncio.sleep(delay=sleep_between_positions)

            board_index += 1


async def evaluate_candidates(
    file_path: Union[str, Path],
    num_positions: int = 1,
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    """This function receives evaluated chess positions from the input queue. It then checks if the
    evaluation of the child board is complete. If not, it sends the child board back to the
    evaluation queue. If the evaluation is complete, it saves the results to the output file.

    Args:
        file_path (Union[str, Path]): The path to the file in which the results are stored.
        num_positions (int, optional): The number of chess positions to evaluate. Defaults to 1.
        sleep_after_get (float, optional): The number of seconds to wait after receiving a board
            from the input queue. Useful to pause this async function and allow other async
            functions to run. Defaults to 0.1.
        identifier_str (str, optional): A string that is used to identify this process.
            Defaults to "".
    """
    # Get the queues
    producer_queue, consumer_queue, _ = connect_to_manager()

    # Create a file to store the results
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    board_counter = 1

    with open(file_path, "a") as file:
        # Create the header of the result file
        csv_header = "parent_fen,child_fen,move,parent_score,child_score\n"
        file.write(csv_header)

        flush_every = 1000

        base_fen: str
        child_fen: Optional[str]
        base_score: Union[Optional[float], str]
        child_score: Union[Optional[float], str]

        while board_counter <= num_positions:
            # Receive data from queue. Use the non-blocking version to avoid
            # blocking the event loop
            while True:
                try:
                    analysis_object: RecommendedMoveAnalysisObject = consumer_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(delay=0.5)
                else:
                    await asyncio.sleep(delay=0.1)
                    break

            await asyncio.sleep(delay=sleep_after_get)

            # We have to handle two different cases:
            # 1. We are analyzing the base board
            #    In this case we need to prepare the result object such that the child board can
            #    be analyzed as well
            # 2. We are analyzing the most promising child board
            #    In this case we are done and can save the base board together with the most
            #    promising child board and the corresponding scores to the output file.
            if not analysis_object.is_complete():  # Case 1
                logging.info(
                    f"[{identifier_str}] Preparing second round for board: " + analysis_object.fen
                )

                # Check that no error occured during the first analysis
                if analysis_object.is_result_valid():
                    # Prepare the result object
                    analysis_object.prepare_second_round()

                    # Add the board to the producer queue
                    producer_queue.put(analysis_object)

            else:  # Case 2
                # Extract the results
                base_fen, base_score, child_fen, child_score, best_move = (
                    analysis_object.parent_fen,
                    analysis_object.parent_score,
                    analysis_object.fen,
                    analysis_object.score,
                    analysis_object.parent_best_move,
                )
                logging.info(f"[{identifier_str}] Saving board {board_counter}: " + base_fen)

                result_str = f"{base_fen},{child_fen},{best_move},{base_score},{child_score}\n"

                # Write the result to the file
                file.write(result_str)

                if board_counter % flush_every == 0:
                    file.flush()
                    os.fsync(file.fileno())

                board_counter += 1

            consumer_queue.task_done()


async def recommended_move_invariance_testing(
    data_generator: BoardGenerator,
    *,
    result_file_path: Optional[Union[str, Path]] = None,
    num_positions: int = 1,
    required_engine_config: Optional[str] = None,
    sleep_after_get: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Main function which starts all the asynchronous tasks and manages the distributed queues.

    Args:
        data_generator (BoardGenerator): A BoardGenerator object that is used to create the
            chess positions.
        result_file_path (Optional[Union[str, Path]], optional): The path to the file in which the
            results are stored. Defaults to None.
        num_positions (int, optional): The number of chess positions to create. Defaults to 1.
        required_engine_config (Optional[str], optional): The name of the engine configuration
            which worker processes should use. Defaults to None.
        sleep_after_get (float, optional): The number of seconds to wait after receiving a board
            from the input queue. Useful to pause async functions and allow other async
            functions to run. Defaults to 0.1.
        logger (Optional[logging.Logger], optional): A logger object that is used to log messages.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    assert result_file_path is not None, "Result file path must be specified"

    # Set up the distributed queues
    engine_queue_in: queue.Queue = queue.Queue()
    engine_queue_out: queue.Queue = queue.Queue()

    def get_input_queue() -> queue.Queue:
        return engine_queue_in

    def get_output_queue() -> queue.Queue:
        return engine_queue_out

    # Initialize the input- and output queues
    if required_engine_config is not None:
        QueueManager.set_engine_config(engine_config=required_engine_config)
    QueueManager.register("input_queue", callable=get_input_queue)
    QueueManager.register("output_queue", callable=get_output_queue)

    net_manager = QueueManager(address=(address, port), authkey=password.encode("utf-8"))

    # Start the server
    net_manager.start()

    # Create all data processing tasks
    data_generator_task = asyncio.create_task(
        create_positions(
            data_generator=data_generator,
            num_positions=num_positions,
            sleep_between_positions=sleep_after_get,
            identifier_str="BOARD_GENERATOR",
        )
    )

    candidate_evaluation_task = asyncio.create_task(
        evaluate_candidates(
            file_path=result_file_path,
            num_positions=num_positions,
            sleep_after_get=sleep_after_get,
            identifier_str="CANDIDATE_EVALUATION",
        )
    )

    # Add callbacks to all tasks
    handle_task_exception = get_task_result_handler(
        logger=logger, message="Task raised an exception"
    )
    for task in [data_generator_task, candidate_evaluation_task]:
        task.add_done_callback(handle_task_exception)

    # Wait for data generator task to finish
    await asyncio.wait([data_generator_task, candidate_evaluation_task])

    # Wait for data queues to become empty
    engine_queue_in.join()
    engine_queue_out.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # fmt: off
    parser.add_argument("--seed",                  type=int, default=42)  # noqa
    parser.add_argument("--engine_config_name",    type=str, default="local_400_nodes.ini")  # noqa
    parser.add_argument("--data_config_name",      type=str, default="database.ini")  # noqa
    parser.add_argument("--num_positions",         type=int, default=1_000_000)  # noqa
    parser.add_argument("--result_subdir",         type=str, default="")  # noqa
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

    # Set random seed
    np.random.seed(args.seed)

    # Create result directory
    config_folder_path = Path(__file__).parent.absolute() / Path("configs/engine_configs/")

    data_config = get_data_generator_config(
        config_name=args.data_config_name,
        config_folder_path=Path(__file__).parent.absolute()
        / Path("configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    # Create results-file-name
    engine_config_name = args.engine_config_name[:-4]
    data_config_name = args.data_config_name[:-4]

    # Build the result file path
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)

    # Store current date and time as string
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")

    result_file_path = result_directory / Path(
        f"results_ENGINE_{engine_config_name}_DATA_{data_config_name}_{dt_string}.txt"
    )

    # Store the experiment configuration in the result file
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        recommended_move_invariance_testing(
            data_generator=data_generator,
            result_file_path=result_file_path,
            num_positions=args.num_positions,
            required_engine_config=args.engine_config_name,
            sleep_after_get=0.1,
            logger=logger,
        )
    )

    end_time = time.perf_counter()
    logging.info(f"Elapsed time: {end_time - start_time: .3f} seconds")
