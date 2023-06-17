import abc
import asyncio
import logging
import os
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


import chess
import chess.engine
import numpy as np

from rl_testing.engine_generators import EngineGenerator
from rl_testing.evolutionary_algorithms.individuals import BoardIndividual, Individual
from rl_testing.util.cache import LRUCache
from rl_testing.util.chess import cp2q, rotate_180_clockwise
from rl_testing.util.engine import RelaxedUciProtocol, engine_analyse
from rl_testing.util.util import get_task_result_handler
from rl_testing.engine_generators.worker import AnalysisObject
from rl_testing.engine_generators.distributed_queue_manager import (
    connect_to_manager,
)

FEN = str


class Fitness(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            (hasattr(subclass, "use_async") and callable(subclass.use_async))
            and (hasattr(subclass, "best_individual") and callable(subclass.best_individual))
            and (hasattr(subclass, "worst_individual") and callable(subclass.worst_individual))
            and (hasattr(subclass, "is_bigger_better"))
            and (
                (hasattr(subclass, "evaluate") and callable(subclass.evaluate))
                or (hasattr(subclass, "evaluate_async") and callable(subclass.evaluate_async))
            )
            or NotImplemented
        )

    @property
    @abc.abstractmethod
    def use_async(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_bigger_better(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def best_individual(self, individuals: List[Individual]) -> Individual:
        raise NotImplementedError

    @abc.abstractmethod
    def worst_individual(self, individuals: List[Individual]) -> Individual:
        raise NotImplementedError

    def evaluate(self, individual: Individual) -> float:
        raise NotImplementedError

    async def evaluate_async(self, individuals: List[Individual]) -> List[float]:
        raise NotImplementedError

    def _find_individual(self, individuals: List[Individual], direction: Callable) -> Individual:
        # Make sure that all individuals have a fitness value and compute it if not.
        for individual in individuals:
            if individual.fitness is None:
                raise ValueError(
                    "Individuals must have a fitness value before calling this method."
                )
                # individual.fitness = self.evaluate(individual)

        fitness_vals = np.array([individual.fitness for individual in individuals])
        return individuals[direction(fitness_vals)]


class PieceNumberFitness(Fitness):
    def __init__(self, more_pieces_better: bool = True) -> None:
        self._more_pieces_better = more_pieces_better

    @property
    def use_async(self) -> bool:
        return False

    @property
    def is_bigger_better(self) -> bool:
        return self._more_pieces_better

    @lru_cache(maxsize=200_000)
    def evaluate(self, board: BoardIndividual) -> float:
        num_pieces = float(len(board.piece_map()))
        return num_pieces if self._more_pieces_better else -num_pieces

    def best_individual(self, individuals: List[BoardIndividual]) -> BoardIndividual:
        return self._find_individual(individuals, np.argmax)

    def worst_individual(self, individuals: List[BoardIndividual]) -> BoardIndividual:
        return self._find_individual(individuals, np.argmin)


class EditDistanceFitness(Fitness):
    def __init__(self, target: str) -> None:
        self._target = self.prepare_fen(target)
        self.distance_cache: dict[Tuple[str, str], int] = {}
        self.max_cache_size = 100000

    def prepare_fen(self, fen: str) -> str:
        return " ".join(fen.split(" ")[:3])

    @property
    def use_async(self) -> bool:
        return False

    @property
    def is_bigger_better(self) -> bool:
        return False

    @lru_cache(maxsize=200_000)
    def evaluate(self, individual: BoardIndividual) -> float:
        if len(self.distance_cache) > self.max_cache_size:
            self.distance_cache: dict[Tuple[str, str], int] = {}
        return self.levenshtein_distance(self._target, self.prepare_fen(individual.fen()))

    def best_individual(self, individuals: List[BoardIndividual]) -> BoardIndividual:
        return self._find_individual(individuals, np.argmin)

    def worst_individual(self, individuals: List[BoardIndividual]) -> BoardIndividual:
        return self._find_individual(individuals, np.argmax)

    def levenshtein_distance(self, string1: str, string2):
        if (string1, string2) in self.distance_cache:
            return self.distance_cache[(string1, string2)]
        if len(string2) == 0:
            return len(string1)
        if len(string1) == 0:
            return len(string2)

        if string1[0] == string2[0]:
            return self.levenshtein_distance(string1[1:], string2[1:])

        distance = min(
            0.5 + self.levenshtein_distance(string1[1:], string2),
            0.5 + self.levenshtein_distance(string1, string2[1:]),
            1 + self.levenshtein_distance(string1[1:], string2[1:]),
        )

        self.distance_cache[(string1, string2)] = distance
        return distance


class BoardSimilarityFitness(Fitness):
    def __init__(self, target: str) -> None:
        self.piece_map = chess.Board(target).piece_map()

    @property
    def use_async(self) -> bool:
        return False

    @property
    def is_bigger_better(self) -> bool:
        return False

    def best_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmin)

    def worst_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmax)

    @lru_cache(maxsize=200_000)
    def evaluate(self, individual: BoardIndividual) -> float:
        fitness = 0.0
        test_piece_map = individual.piece_map()

        for square in chess.SQUARES:
            if square in self.piece_map and square not in test_piece_map:
                fitness += 0.5
            elif square not in self.piece_map and square in test_piece_map:
                fitness += 1.0
            elif square in self.piece_map and square in test_piece_map:
                if self.piece_map[square].color != test_piece_map[square].color:
                    fitness += 1.0
                elif self.piece_map[square].piece_type != test_piece_map[square].piece_type:
                    fitness += 0.5

        return fitness


class HashFitness(Fitness):
    @property
    def use_async(self) -> bool:
        return False

    @property
    def is_bigger_better(self) -> bool:
        return True

    def best_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmax)

    def worst_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmin)

    @lru_cache(maxsize=200_000)
    def evaluate(self, individual: BoardIndividual) -> float:
        assert hasattr(individual, "__hash__"), "Individual must have implemented a hash method"
        return hash(individual)


class DifferentialTestingFitness(Fitness):
    @staticmethod
    async def setup_engine(
        engine_generator: EngineGenerator, network_name: Optional[str]
    ) -> RelaxedUciProtocol:
        if network_name is not None:
            engine_generator.set_network(network_name)

        return await engine_generator.get_initialized_engine()

    @staticmethod
    async def analyze_positions(
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        engine_generator: EngineGenerator,
        search_limits: Dict[str, Any],
        sleep_after_get: float = 0.0,
        network_name: Optional[str] = None,
        identifier_str: str = "",
    ) -> None:
        """Runs an infinite loop that fetches boards from the input queue, analyzes them using an engine running either locally
        or remotely, and puts the results in the output queue.

        Args:
            input_queue (asyncio.Queue[chess.Board]): The queue from which to fetch boards.
            output_queue (asyncio.Queue[Union[Tuple[FEN, chess.Move, float], Tuple[FEN, str, str]]]): The queue to which to put the results.
            engine_generator (EngineGenerator): The generator to create the engines.
            search_limits (Dict[str, Any]): The search limits which the engine uses for the analysis.
            sleep_after_get (float, optional): How long to sleep after fetching a board from the input queue. Defaults to 0.0.
            network_name (Optional[str], optional): The name of the network to use. Defaults to None.
            identifier_str (str, optional): A string to identify this particular engine. Defaults to "".
        """
        invalid_placeholder = ["invalid", "invalid"]
        nan_placeholder = ["nan", "nan"]

        # Setup the engine
        engine = await DifferentialTestingFitness.setup_engine(engine_generator, network_name)

        # Required to ensure that the engine doesn't use cached results from
        # previous analyses
        analysis_counter = 0

        while True:
            # Fetch the next board from the queue
            board_index, board = await input_queue.get()
            fen = board.fen()
            await asyncio.sleep(delay=sleep_after_get)

            if (board_index + 1) % 10 == 0:
                logging.info(f"[{identifier_str}] Analyzing board {board_index + 1}: {fen}")

            # Needs to be in a try-except because the engine might crash unexpectedly
            try:
                analysis_counter += 1
                info = await engine_analyse(
                    engine,
                    chess.Board(fen),
                    chess.engine.Limit(**search_limits),
                    game=analysis_counter,
                    intermediate_info=False,
                )

            except chess.engine.EngineTerminatedError:
                # Mark the current board as failed
                await output_queue.put((fen, *invalid_placeholder))

                # Try to kill the failed engine
                logging.info(f"[{identifier_str}] Trying to kill engine")
                engine_generator.kill_engine(engine=engine)

                # Try to restart the engine
                logging.info(f"[{identifier_str}] Trying to restart engine")

                engine = await DifferentialTestingFitness.setup_engine(
                    engine_generator, network_name
                )

            else:
                score_cp = info["score"].relative.score(mate_score=12780)

                # Check if the computed score is valid
                if engine_generator is not None and not engine_generator.cp_score_valid(score_cp):
                    await output_queue.put((fen, *nan_placeholder))

                # Check if the proposed best move is valid
                elif engine.invalid_best_move:
                    await output_queue.put((fen, *invalid_placeholder))
                else:
                    best_move = info["pv"][0]
                    await output_queue.put((fen, best_move, cp2q(score_cp)))
            finally:
                input_queue.task_done()

    @staticmethod
    async def write_output(
        input_queue: asyncio.Queue,
        result_file_path: str,
        identifier_str: str = "",
    ) -> None:
        buffer_limit = 1000
        with open(result_file_path, "r+") as result_file:
            buffer_size = 0
            config_str = result_file.read()

            result_file.write("fen,fitness\n")
            while True:
                buffer_size += 1
                fen, fitness = await input_queue.get()
                result_file.write(f"{fen},{fitness}\n")
                if buffer_size > 0 and buffer_size % buffer_limit == 0:
                    logging.info(f"[{identifier_str}] Write {buffer_limit} results to file")
                    result_file.flush()
                    os.fsync(result_file.fileno())
                    buffer_size = 0

                input_queue.task_done()

    def __init__(
        self,
        engine_generator1: EngineGenerator,
        engine_generator2: EngineGenerator,
        search_limits1: Dict[str, Any],
        search_limits2: Dict[str, Any],
        network_name1: Optional[str] = None,
        network_name2: Optional[str] = None,
        num_engines1: int = 1,
        num_engines2: int = 1,
        result_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the DifferentialTestingFitness class.

        Args:
            engine_generator1 (EngineGenerator): A generator for the first engine.
            engine_generator2 (EngineGenerator): A generator for the second engine.
            search_limits1 (Dict[str, Any]): A dictionary of search limits for the first engine.
            search_limits2 (Dict[str, Any]): A dictionary of search limits for the second engine.
            network_name1 (Optional[str], optional): An optional name for the network used by the first engine. Defaults to None.
            network_name2 (Optional[str], optional): An optional name for the network used by the second engine. Defaults to None.
            num_engines1 (int, optional): How many instances of the first engine to use. Defaults to 1.
            num_engines2 (int, optional): How many instances of the second engine to use. Defaults to 1.
        """
        # Create a logger if it doesn't exist
        self.logger = logger or logging.getLogger(__name__)

        # Initialize all the variables
        self.engine_generator1 = engine_generator1
        self.engine_generator2 = engine_generator2
        self.search_limits1 = search_limits1
        self.search_limits2 = search_limits2
        self.network_name1 = network_name1
        self.network_name2 = network_name2
        self.num_engines1 = num_engines1
        self.num_engines2 = num_engines2
        self.result_path = result_path
        self.input_queue1 = asyncio.Queue()
        self.input_queue2 = asyncio.Queue()
        self.output_queue1 = asyncio.Queue()
        self.output_queue2 = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.cache: Dict[FEN, float] = LRUCache(maxsize=200_000)

        self.engine_tasks: List[asyncio.Task] = []
        self.result_task = None

        # Log how many times a position has been truly evaluated (not cached)
        self.num_evaluations = 0

    async def create_tasks(self) -> None:
        handle_task_exception = get_task_result_handler(
            logger=self.logger, message="Task raised an exception"
        )
        for (
            group_index,
            num_engines_to_create,
            input_queue,
            output_queue,
            engine_generator,
            network_name,
            search_limits,
        ) in zip(
            [1, 2],
            [self.num_engines1, self.num_engines2],
            [self.input_queue1, self.input_queue2],
            [self.output_queue1, self.output_queue2],
            [self.engine_generator1, self.engine_generator2],
            [self.network_name1, self.network_name2],
            [self.search_limits1, self.search_limits2],
        ):
            for engine_index in range(num_engines_to_create):
                self.engine_tasks.append(
                    asyncio.create_task(
                        DifferentialTestingFitness.analyze_positions(
                            input_queue=input_queue,
                            output_queue=output_queue,
                            engine_generator=engine_generator,
                            network_name=network_name,
                            search_limits=search_limits,
                            sleep_after_get=0.1,
                            identifier_str=f"GROUP {group_index}, ENGINE {engine_index+1}",
                        )
                    )
                )
                # Add a callback to handle exceptions
                self.engine_tasks[-1].add_done_callback(handle_task_exception)

        # Create the task for writing the results to file
        if self.result_path is not None:
            self.result_task = asyncio.create_task(
                DifferentialTestingFitness.write_output(
                    input_queue=self.result_queue,
                    result_file_path=self.result_path,
                    identifier_str="RESULT WRITER",
                )
            )

            self.result_task.add_done_callback(handle_task_exception)

    def cancel_tasks(self) -> None:
        """Cancels all the tasks."""
        # First kill all running engine processes
        self.engine_generator1.kill_all_engines()
        self.engine_generator2.kill_all_engines()

        # Then cancel all the tasks
        for task in self.engine_tasks:
            task.cancel()

        if self.result_task is not None:
            self.result_task.cancel()

    @property
    def use_async(self) -> bool:
        return True

    @property
    def is_bigger_better(self) -> bool:
        return True

    def best_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmax)

    def worst_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmin)

    async def evaluate_async(self, individuals: List[BoardIndividual]) -> List[float]:
        """Evaluates the given individuals asynchronously.

        Args:
            individuals: The individuals to evaluate.

        Returns:
            The fitness values of the individuals.
        """
        # A dictionary to store fens which are currently being processed together with their positions in the results list
        fens_in_progress: Dict[FEN, List[int]] = {}

        # Prepare the result list and fill it with a negative value. This fitness function only
        # produces positive values, so this is a good way to mark invalid individuals.
        results: List[float] = [-1.0] * len(individuals)

        # Iterate over the individuals and either compute their fitness or fetch the fitness from the cache
        for index, individual in enumerate(individuals):
            fen: FEN = individual.fen()
            if fen in self.cache:
                results[index] = self.cache[fen]
            elif fen not in fens_in_progress:
                fens_in_progress[fen] = [index]
                await self.input_queue1.put((index, individual))
                await self.input_queue2.put((index, individual))
                self.num_evaluations += 1
            else:
                fens_in_progress[fen].append(index)

        # Wait until all boards have been processed
        await self.input_queue1.join()
        await self.input_queue2.join()

        # An output dictionary to match the results of the two output queues
        output_dict: Dict[FEN, float] = {}

        # Extract all results from the first output queue
        while not self.output_queue1.empty():
            fen, _, score = await self.output_queue1.get()
            output_dict[fen] = score
            self.output_queue1.task_done()

        # Extract all results from the second output queue and compute the score difference
        while not self.output_queue2.empty():
            fen, _, score = await self.output_queue2.get()

            # Both results are valid
            if (output_dict[fen] != "invalid" and score != "invalid") and (
                output_dict[fen] != "nan" and score != "nan"
            ):
                fitness = abs(output_dict[fen] - score)

                # Add the fitness value to all individuals with the same fen
                for index in fens_in_progress[fen]:
                    results[index] = fitness

                # Add the fitness value to the cache
                self.cache[fen] = fitness
            elif output_dict[fen] == "invalid" or score == "invalid":
                # Cache the invalid value anyway to prevent future re-computations (and crashes) of the same board
                self.cache[fen] = -1.0
            elif output_dict[fen] == "nan" or score == "nan":
                self.cache[fen] = -2.0

            # No matter the outcome, write the result to the result file
            if self.result_path is not None:
                await self.result_queue.put((fen, self.cache[fen]))

            self.output_queue2.task_done()

        # Wait until all results have been written to file
        if self.result_path is not None:
            await self.result_queue.join()

        return results


def less_pieces_fitness(individual: chess.Board) -> float:
    """A fitness function which rewards individuals with less pieces on the board.
    The fitness is normalized to lie in the interval [0, 0.5] where a higher value means a better fitness.

    Args:
        individual: The individual to evaluate.

    Returns:
        The fitness value of the individual.
    """
    # The minimum number of possible pieces is 2 (two kings)
    # The maximum number of possible pieces is 32 (16 white and 16 black pieces)
    return (1.0 - (len(individual.piece_map()) - 2) / 30.0) / 2


class CrashTestingFitness(DifferentialTestingFitness):
    async def evaluate_async(self, individuals: List[BoardIndividual]) -> List[float]:
        """Evaluates the given individuals asynchronously.

        Args:
            individuals: The individuals to evaluate.

        Returns:
            The fitness values of the individuals.
        """
        # A dictionary to store fens which are currently being processed together with their positions in the results list
        fens_in_progress: Dict[FEN, List[int]] = {}

        # Prepare the result list and fill it with a negative value. This fitness function only
        # produces positive values, so this is a good way to mark invalid individuals.
        results: List[float] = [-1.0] * len(individuals)

        # Iterate over the individuals and either compute their fitness or fetch the fitness from the cache
        for index, individual in enumerate(individuals):
            fen: FEN = individual.fen()
            if fen in self.cache:
                results[index] = self.cache[fen]
            elif fen not in fens_in_progress:
                fens_in_progress[fen] = [index]
                await self.input_queue1.put((index, individual))
                await self.input_queue2.put((index, individual))
                self.num_evaluations += 1
            else:
                fens_in_progress[fen].append(index)

        # Wait until all boards have been processed
        await self.input_queue1.join()
        await self.input_queue2.join()

        # An output dictionary to match the results of the two output queues
        output_dict: Dict[FEN, float] = {}

        # Extract all results from the first output queue
        while not self.output_queue1.empty():
            fen, _, score = await self.output_queue1.get()
            output_dict[fen] = score
            self.output_queue1.task_done()

        # Extract all results from the second output queue and compute the score difference
        while not self.output_queue2.empty():
            fen, _, score = await self.output_queue2.get()
            board = chess.Board(fen)

            # Both results are valid
            if (output_dict[fen] != "invalid" and score != "invalid") and (
                output_dict[fen] != "nan" and score != "nan"
            ):
                fitness = abs(output_dict[fen] - score) + less_pieces_fitness(board)
            elif output_dict[fen] == "invalid" or score == "invalid":
                # Cache the invalid value anyway to prevent future re-computations (and crashes) of the same board
                fitness = 2.0 + less_pieces_fitness(board)
            elif output_dict[fen] == "nan" or score == "nan":
                fitness = 2.0 + less_pieces_fitness(board)

            # Add the fitness value to all individuals with the same fen
            for index in fens_in_progress[fen]:
                results[index] = fitness

            # Add the fitness value to the cache
            self.cache[fen] = fitness

            # No matter the outcome, write the result to the result file
            if self.result_path is not None:
                await self.result_queue.put((fen, self.cache[fen]))

            self.output_queue2.task_done()

        # Wait until all results have been written to file
        if self.result_path is not None:
            await self.result_queue.join()

        return results


class BoardTransformationFitness(Fitness):
    @staticmethod
    async def write_output(
        input_queue: asyncio.Queue,
        result_file_path: str,
        identifier_str: str = "",
    ) -> None:
        buffer_limit = 1000
        with open(result_file_path, "r+") as result_file:
            buffer_size = 0
            _ = result_file.read()

            result_file.write("fen1,fitness1,fen2,fitness2\n")
            while True:
                buffer_size += 1
                fen1, fitness1, fen2, fitness2 = await input_queue.get()
                result_file.write(f"{fen1},{fitness1},{fen2},{fitness2}\n")
                if buffer_size > 0 and buffer_size % buffer_limit == 0:
                    logging.info(f"[{identifier_str}] Write {buffer_limit} results to file")
                    result_file.flush()
                    os.fsync(result_file.fileno())
                    buffer_size = 0

                input_queue.task_done()

    def __init__(
        self,
        result_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the BoardTransformationFitness class.

        Args:
            result_path (Optional[str], optional): The path to the file where the results should be written. Defaults to None.
            logger (Optional[logging.Logger], optional): A logger to use. Defaults to None.
        """
        # Create a logger if it doesn't exist
        self.logger = logger or logging.getLogger(__name__)

        # Initialize all the variables
        self.result_path = result_path
        self.input_queue, self.output_queue, _ = connect_to_manager()
        self.result_queue = asyncio.Queue()
        self.cache: Dict[FEN, float] = LRUCache(maxsize=200_000)

        self.result_task = None

        # Log how many times a position has been truly evaluated (not cached)
        self.num_evaluations = 0

    async def create_tasks(self) -> None:
        handle_task_exception = get_task_result_handler(
            logger=self.logger, message="Task raised an exception"
        )

        # Create the task for writing the results to file
        if self.result_path is not None:
            self.result_task = asyncio.create_task(
                BoardTransformationFitness.write_output(
                    input_queue=self.result_queue,
                    result_file_path=self.result_path,
                    identifier_str="RESULT WRITER",
                )
            )

            self.result_task.add_done_callback(handle_task_exception)

    def cancel_tasks(self) -> None:
        """Cancels all the tasks."""

        if self.result_task is not None:
            self.result_task.cancel()

    @property
    def use_async(self) -> bool:
        return True

    @property
    def is_bigger_better(self) -> bool:
        return True

    def best_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmax)

    def worst_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmin)

    async def evaluate_async(self, individuals: List[BoardIndividual]) -> List[float]:
        """Evaluates the given individuals asynchronously.

        Args:
            individuals: The individuals to evaluate.

        Returns:
            The fitness values of the individuals.
        """
        # A dictionary to store fens which are currently being processed together with their positions in the results list
        fens_in_progress: Set[FEN] = set()

        # Prepare the result list and fill it with a negative value. This fitness function only
        # produces positive values, so this is a good way to mark invalid individuals.
        results: List[float] = [-1.0] * len(individuals)
        result_fens: List[Tuple[FEN, FEN]] = []

        # An output dictionary to match the results of the two output queues
        output_dict: Dict[FEN, float] = {}

        print(f"Before processing: {len(individuals)})")

        # Iterate over the individuals and either compute their fitness or fetch the fitness from the cache
        for index, individual in enumerate(individuals):
            fen1: FEN = individual.fen()
            fen2: FEN = chess.Board(fen1).transform(rotate_180_clockwise).fen()
            result_fens.append((fen1, fen2))

            for fen in [fen1, fen2]:
                if fen in self.cache:
                    output_dict[fen] = self.cache[fen]
                elif fen not in fens_in_progress:
                    fens_in_progress.add(fen)
                    self.input_queue.put(AnalysisObject(fen=fen))
                    self.num_evaluations += 1

        # Wait until all boards have been processed
        self.input_queue.join()

        # Extract all results from the first output queue
        while not self.output_queue.empty():
            analysis_object: AnalysisObject = self.output_queue.get()
            output_dict[analysis_object.fen] = analysis_object.score
            self.output_queue.task_done()

        # Extract all results from the second output queue and compute the score difference
        for index, (fen1, fen2) in enumerate(result_fens):
            # Both results are valid
            if output_dict[fen1] not in ["invalid", "nan", None] and output_dict[fen2] not in [
                "invalid",
                "nan",
                None,
            ]:
                fitness = abs(output_dict[fen1] - output_dict[fen2])

                results[index] = fitness

                # Add the scores to the cache
                self.cache[fen1] = output_dict[fen1]
                self.cache[fen2] = output_dict[fen2]

            # No matter the outcome, write the result to the result file
            if self.result_path is not None:
                await self.result_queue.put((fen1, self.cache[fen1], fen2, self.cache[fen2]))

        # Wait until all results have been written to file
        if self.result_path is not None:
            await self.result_queue.join()

        return results


if __name__ == "__main__":
    # board1 = chess.Board("8/1p6/1p6/pPp1p1n1/P1P1P1k1/1K1P4/8/2B5 w - - 110 118")
    # board2 = chess.Board("r3qb1r/pppbk1p1/2np2np/4p2Q/2BPP3/2P5/PP3PPP/RNB2RK1 w - - 4 11")
    # fitness = PieceNumberFitness()

    # print("board1: ", fitness.evaluate(board1))
    # print("board2: ", fitness.evaluate(board2))
    pass
