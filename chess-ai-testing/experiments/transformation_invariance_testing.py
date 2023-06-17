import argparse
import asyncio
import logging
import os
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import chess
import chess.engine
import numpy as np
from chess import flip_anti_diagonal, flip_diagonal, flip_horizontal, flip_vertical

from rl_testing.config_parsers import get_data_generator_config
from rl_testing.data_generators import BoardGenerator, get_data_generator

from rl_testing.engine_generators.distributed_queue_manager import (
    QueueManager,
    address,
    connect_to_manager,
    password,
    port,
)
from rl_testing.engine_generators.worker import TransformationAnalysisObject
from rl_testing.util.chess import apply_transformation
from rl_testing.util.chess import remove_pawns as remove_pawns_func
from rl_testing.util.chess import (
    rotate_90_clockwise,
    rotate_180_clockwise,
    rotate_270_clockwise,
)
from rl_testing.util.experiment import store_experiment_params
from rl_testing.util.util import get_task_result_handler

RESULT_DIR = Path(__file__).parent / Path("results/transformation_testing")

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


class ReceiverCache:
    """This class is used to cache the received data until all the data for a board has been
    received.
    """

    def __init__(self, consumer_queue: queue.Queue, num_transformations: int) -> None:
        """Initializes the ReceiverCache object.

        Args:
            consumer_queue (queue.Queue): A queue from which the data is received.
            num_transformations (int): The number of transformations that are applied to each
                board.
        """
        # Print type of consumer_queue
        self.consumer_queue = consumer_queue
        self.num_transformations = num_transformations

        self.score_cache: Dict[str, List[Optional[float]]] = {}

    async def receive_data(self) -> List[Iterable[Any]]:
        """This function repeatedly fetches data from the queue until all the data for a single
        board has been received.

        Returns:
            List[Iterable[Any]]: A list of tuples containing the board's fen and the scores for
                the board and its transformations.
        """
        # Receive data from queue
        while True:
            try:
                analysis_object: TransformationAnalysisObject = self.consumer_queue.get_nowait()
                base_fen = analysis_object.base_fen
                transform_index = analysis_object.transformation_index
                score = analysis_object.score
            except queue.Empty:
                await asyncio.sleep(delay=0.5)
            else:
                await asyncio.sleep(delay=0.1)
                break

        # The boards might not arrive in the correct order due to the asynchronous nature of
        # the program. Therefore, we need to cache the boards and scores until we have all
        # of them.

        if base_fen in self.score_cache:
            self.score_cache[base_fen][transform_index] = score
        else:
            self.score_cache[base_fen] = [None] * self.num_transformations
            self.score_cache[base_fen][transform_index] = score

        complete_data_tuples = []
        # Check if we have all the data for this board
        if all([element is not None for element in self.score_cache[base_fen]]):
            # We have all the data for this board
            complete_data_tuples.append((base_fen, self.score_cache[base_fen]))
            del self.score_cache[base_fen]

        return complete_data_tuples


async def create_positions(
    data_generator: BoardGenerator,
    transformation_functions: List[Callable[[chess.Bitboard], chess.Bitboard]],
    remove_pawns: bool = False,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    identifier_str: str = "",
) -> None:
    """Create chess positions using the provided data generator, apply all the transformations
    specified in the transformation_functions list and send the results to the output queue.

    Args:
        data_generator (BoardGenerator): A BoardGenerator object that is used to create the
            chess positions.
        transformation_functions (List[Callable[[chess.Bitboard], chess.Bitboard]]): A list of
            functions that are used to transform the chess positions.
        remove_pawns (bool, optional): Whether or not to remove the pawns from the generated chess
            positions. Defaults to False.
        num_positions (int, optional): The number of chess positions to create. Defaults to 1.
        sleep_between_positions (float, optional): The number of seconds to wait between creating
            two chess positions. Useful to pause this async function and allow other async
            functions to run. Defaults to 0.1.
        identifier_str (str, optional): A string that is used to identify this process.
            Defaults to "".
    """
    fen_cache = {}

    # Get the queues
    output_queue: queue.Queue
    output_queue, _, _ = connect_to_manager()

    # Create the chess positions
    board_index = 1
    while board_index <= num_positions:
        # Create a random chess position
        board_candidate = data_generator.next()

        if board_candidate != "failed" and remove_pawns:
            board_candidate = remove_pawns_func(board_candidate)

        # Check if the generated position was valid
        if board_candidate != "failed" and board_candidate.fen() not in fen_cache:
            fen_cache[board_candidate.fen()] = True

            # Apply the transformations to the board
            transformed_boards = [board_candidate]
            for transformation_function in transformation_functions:
                transformed_boards.append(
                    apply_transformation(board_candidate, transformation_function)
                )
            fen = board_candidate.fen(en_passant="fen")
            logging.info(f"[{identifier_str}] Created base board {board_index + 1}: {fen}")

            # Log the transformed boards
            for transformed_board in transformed_boards[1:]:
                fen = transformed_board.fen(en_passant="fen")
                logging.info(f"[{identifier_str}] Created transformed board: {fen}")

            # Send the boards to the output queue
            for transform_index, transformed_board in enumerate(transformed_boards):
                analysis_object = TransformationAnalysisObject(
                    fen=transformed_board.fen(en_passant="fen"),
                    base_fen=board_candidate.fen(en_passant="fen"),
                    transformation_index=transform_index,
                )
                output_queue.put(analysis_object)

            await asyncio.sleep(delay=sleep_between_positions)

            board_index += 1


async def evaluate_candidates(
    num_transforms: int,
    file_path: Union[str, Path],
    num_positions: int = 1,
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:
    """This function receives the evaluated chess positions from the input queue and writes them
    to a file.

    Args:
        num_transforms (int): The number of transformations that are applied to each board.
        file_path (Union[str, Path]): The path to the file in which the results are stored.
        num_positions (int, optional): The number of chess positions to evaluate. Defaults to 1.
        sleep_after_get (float, optional): The number of seconds to wait after receiving a board
            from the input queue. Useful to pause this async function and allow other async
            functions to run. Defaults to 0.1.
        identifier_str (str, optional): A string that is used to identify this process.
            Defaults to "".
    """
    # Get the queues
    engine_queue: queue.Queue
    _, engine_queue, _ = connect_to_manager()

    # Create a file to store the results
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    board_counter = 1

    # Initialize the receiver cache
    receiver_cache = ReceiverCache(consumer_queue=engine_queue, num_transformations=num_transforms)

    with open(file_path, "a") as file:
        flush_every = 1000
        while board_counter <= num_positions:
            # Fetch the next board and the corresponding scores from the queues
            complete_data_tuples = await receiver_cache.receive_data()

            # Iterate over the received data
            for fen, scores in complete_data_tuples:
                logging.info(f"[{identifier_str}] Saving board {board_counter}: " + fen)

                # Write the found adversarial example into a file
                result_str = f"{fen},"
                for score in scores:
                    # Add the score to the result string
                    result_str += f"{score},"

                    # Mark the element as processed
                    engine_queue.task_done()

                result_str = result_str[:-1] + "\n"

                # Write the result to the file
                file.write(result_str)

                if board_counter % flush_every == 0:
                    file.flush()
                    os.fsync(file.fileno())

                board_counter += 1


async def transformation_invariance_testing(
    data_generator: BoardGenerator,
    transformation_functions: List[Callable[[chess.Bitboard], chess.Bitboard]],
    *,
    result_file_path: Optional[Union[str, Path]] = None,
    remove_pawns: bool = False,
    num_positions: int = 1,
    required_engine_config: Optional[str] = None,
    sleep_after_get: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Main function which starts all the asynchronous tasks and manages the distributed queues.

    Args:
        data_generator (BoardGenerator): A BoardGenerator object that is used to create the
            chess positions.
        transformation_functions (List[Callable[[chess.Bitboard], chess.Bitboard]]): A list of
            functions that are used to transform the chess positions.
        result_file_path (Optional[Union[str, Path]], optional): The path to the file in which the
            results are stored. Defaults to None.
        remove_pawns (bool, optional): Whether or not to remove the pawns from the generated chess
            positions. Defaults to False.
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
            transformation_functions=transformation_functions,
            remove_pawns=remove_pawns,
            num_positions=num_positions,
            sleep_between_positions=sleep_after_get,
            identifier_str="BOARD_GENERATOR",
        )
    )

    candidate_evaluation_task = asyncio.create_task(
        evaluate_candidates(
            num_transforms=len(transformation_functions) + 1,
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
    for task in [
        data_generator_task,
        candidate_evaluation_task,
    ]:
        task.add_done_callback(handle_task_exception)

    # Wait for data generator task to finish
    await asyncio.wait([data_generator_task, candidate_evaluation_task])

    # Wait for data queues to become empty
    engine_queue_in.join()
    engine_queue_out.join()

    # Cancel all remaining tasks
    # candidate_evaluation_task.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # fmt: off
    parser.add_argument("--seed",                           type=int, default=42)  # noqa
    parser.add_argument("--engine_config_name",             type=str, default="local_400_nodes.ini")  # noqa
    parser.add_argument("--data_config_name",               type=str, default="database.ini")  # noqa
    parser.add_argument("--remove_pawns",                   action="store_true")  # noqa
    parser.add_argument("--num_positions",                  type=int, default=1_000_000)  # noqa
    parser.add_argument("--transformations",                type=str, default=["rot90", "rot180", "rot270", "flip_diag", "flip_anti_diag", "flip_hor", "flip_vert"], nargs="+",  # noqa
                                                            choices=["rot90", "rot180", "rot270", "flip_diag", "flip_anti_diag", "flip_hor", "flip_vert", "mirror"])  # noqa
    parser.add_argument("--result_subdir",                  type=str, default="")  # noqa
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

    # Build the data generator
    data_config = get_data_generator_config(
        config_name=args.data_config_name,
        config_folder_path=Path(__file__).parent.absolute()
        / Path("configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    # Extract the transformations
    transformation_functions = [
        transformation_dict[transformation_name] for transformation_name in args.transformations
    ]

    # Create results-file-name
    data_config_name = args.data_config_name[:-4]

    # Build the result file path
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)

    # Store current date and time as string
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")

    result_file_path = result_directory / Path(
        f"results_ENGINE_{args.engine_config_name[:-4]}_DATA_{data_config_name}_{dt_string}.txt"
    )

    # Store the experiment configuration in the result file
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Store the transformation names in the result file
    with open(result_file_path, "a") as result_file:
        result_file.write("fen,original,")
        transformation_str = "".join(
            [f"{transformation}," for transformation in args.transformations]
        )
        result_file.write(f"{transformation_str[:-1]}\n")

    # Extract the boolean parameter
    remove_pawns = args.remove_pawns
    if remove_pawns is None:
        remove_pawns = False

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        transformation_invariance_testing(
            data_generator=data_generator,
            transformation_functions=transformation_functions,
            result_file_path=result_file_path,
            remove_pawns=remove_pawns,
            num_positions=args.num_positions,
            required_engine_config=args.engine_config_name,
            sleep_after_get=0.1,
            logger=logger,
        )
    )

    end_time = time.perf_counter()
    logging.info(f"Elapsed time: {end_time - start_time: .3f} seconds")
