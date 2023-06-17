import argparse
import asyncio
import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess.engine
import numpy as np

from evolutionary_algorithm_configs import (
    SimpleEvolutionaryAlgorithmConfig,
)
from rl_testing.evolutionary_algorithms import (
    get_initialized_crossover,
    get_initialized_mutator,
    get_initialized_selector,
)
from rl_testing.evolutionary_algorithms.algorithms import AsyncEvolutionaryAlgorithm
from rl_testing.evolutionary_algorithms.crossovers import Crossover
from rl_testing.evolutionary_algorithms.fitnesses import (
    BoardSimilarityFitness,
    DifferentialTestingFitness,
    BoardTransformationFitness,
    EditDistanceFitness,
    Fitness,
    HashFitness,
    PieceNumberFitness,
)
from rl_testing.evolutionary_algorithms.mutations import Mutator
from rl_testing.evolutionary_algorithms.populations import SimplePopulation
from rl_testing.evolutionary_algorithms.selections import (
    Selector,
)
from rl_testing.evolutionary_algorithms.statistics import SimpleStatistics
from rl_testing.util.cache import LRUCache
from rl_testing.util.evolutionary_algorithm import (
    get_random_individuals,
    should_decrease_probability,
)
from rl_testing.util.experiment import (
    get_experiment_params_dict,
)
from rl_testing.util.util import get_random_state, log_time

from rl_testing.engine_generators.distributed_queue_manager import (
    QueueManager,
    address,
    port,
    password,
)
import queue

RESULT_DIR = Path(__file__).parent.parent / Path("results/evolutionary_algorithm")
CONFIG_FOLDER = Path(__file__).parent.parent
WANDB_CONFIG_FILE = CONFIG_FOLDER / Path(
    "configs/evolutionary_algorithm_configs/config_ea_differential_testing.yaml"
)
ENGINE_CONFIG_FOLDER = CONFIG_FOLDER / Path("configs/engine_configs")
EVOLUTIONARY_ALGORITHM_CONFIG_FOLDER = CONFIG_FOLDER / Path(
    "configs/evolutionary_algorithm_configs"
)
DEBUG = True
Time = float


class DistributedOracleQueryEvolutionaryAlgorithm(AsyncEvolutionaryAlgorithm):
    """Evolutionary algorithm which allows for distributed oracle queries."""

    def __init__(
        self,
        evolutionary_algorithm_config: SimpleEvolutionaryAlgorithmConfig,
        experiment_config: Dict[str, Any],
        logger: logging.Logger,
        result_file_path: Optional[Path] = None,
    ):
        """Initialize the evolutionary algorithm.

        Args:
            evolutionary_algorithm_config (SimpleEvolutionaryAlgorithmConfig): The evolutionary
                algorithm config.
            experiment_config (Dict[str, Any]): The experiment config.
            logger (logging.Logger): The logger.
            result_file_path (Optional[Path], optional): The path to the file where the results
                should be stored. Defaults to None.
        """
        # Experiment configs
        self.experiment_config = experiment_config
        self.evolutionary_algorithm_config = evolutionary_algorithm_config
        self.logger = logger

        # Evolutionary algorithm configs
        self.data_path = evolutionary_algorithm_config.data_path
        self.num_generations = evolutionary_algorithm_config.num_generations
        self.crossover_probability = evolutionary_algorithm_config.crossover_probability
        self.mutation_probability = evolutionary_algorithm_config.mutation_probability
        self.early_stopping = evolutionary_algorithm_config.early_stopping
        self.early_stopping_value = evolutionary_algorithm_config.early_stopping_value
        self.probability_decay = evolutionary_algorithm_config.probability_decay

        # Fitness function configs
        self.max_num_fitness_evaluations = evolutionary_algorithm_config.max_num_fitness_evaluations
        self._num_fitness_evaluations = 0
        self.fitness: Optional[BoardTransformationFitness] = None
        self.fitness_cache: Optional[LRUCache] = None
        self.result_file_path = result_file_path
        self.input_queue: queue.Queue = queue.Queue()
        self.output_queue: queue.Queue = queue.Queue()

        # Prepare the queues for the distributed fitness evaluation
        def get_input_queue() -> queue.Queue:
            return self.input_queue

        def get_output_queue() -> queue.Queue:
            return self.output_queue

        # Initialize the input- and output queues
        QueueManager.register("input_queue", callable=get_input_queue)
        QueueManager.register("output_queue", callable=get_output_queue)

        self.net_manager = QueueManager(address=(address, port), authkey=password.encode("utf-8"))

        # Start the server
        self.net_manager.start()

    async def initialize(self, seed: int) -> None:
        """Initialize the evolutionary algorithm by creating the random state, the multiprocessing
        pool, the fitness function, the evolutionary operators and the population.

        Args:
            seed (int): The seed for the random state.
        """
        # Create the random state
        self.random_state = get_random_state(seed)

        # Create a multiprocessing pool
        self.pool = multiprocessing.Pool(processes=self.evolutionary_algorithm_config.num_workers)

        # Create the fitness function
        self.fitness = BoardTransformationFitness(
            **self.experiment_config["fitness_config"],
            # input_queue=self.input_queue,
            # output_queue=self.output_queue,
            result_path=self.result_file_path,
            logger=self.logger,
        )

        # Reuse the fitness cache if it exists
        if self.fitness_cache is not None:
            self.fitness.cache = self.fitness_cache

        # Create the evolutionary operators
        self.mutate: Mutator = get_initialized_mutator(self.evolutionary_algorithm_config)
        self.crossover: Crossover = get_initialized_crossover(self.evolutionary_algorithm_config)
        self.select: Selector = get_initialized_selector(self.evolutionary_algorithm_config)

        # Create the population
        individuals = get_random_individuals(
            self.data_path,
            self.evolutionary_algorithm_config.population_size,
            self.random_state,
        )
        self.population = SimplePopulation(
            individuals=individuals,
            fitness=self.fitness,
            mutator=self.mutate,
            crossover=self.crossover,
            selector=self.select,
            pool=self.pool,
            _random_state=self.random_state,
        )

        await self.fitness.create_tasks()

    @property
    def num_fitness_evaluations(self) -> int:
        """Return the number of fitness evaluations.

        Returns:
            int: The number of fitness evaluations.
        """
        if self.fitness is None:
            return self._num_fitness_evaluations
        return len(self.fitness.cache)

    async def run(self) -> SimpleStatistics:
        """Runs the evolutionary algorithm until an early stopping criterion is reached or the
        maximum number of fitness evaluations is reached.

        Returns:
            SimpleStatistics: The statistics of the evolutionary algorithm.
        """
        start_time = time.time()

        # Create the statistics
        statistics = SimpleStatistics()

        # Check if the maximum number of fitness evaluations is reached
        if self.num_fitness_evaluations >= self.max_num_fitness_evaluations:
            return statistics

        # Evaluate the entire population
        await self.population.evaluate_individuals_async()

        # Check if the maximum number of fitness evaluations is reached
        if self.num_fitness_evaluations >= self.max_num_fitness_evaluations:
            statistics.fill_time_series(0, self.num_generations, self.population)
            logging.info("Early stopping!")
            return statistics

        for generation in range(self.num_generations):
            logging.info(f"\n\nGeneration {generation}")

            # Select the next generation individuals
            log_time(start_time, "before selecting")
            self.population.create_next_generation()

            # Apply crossover on the offspring
            log_time(start_time, "before mating")
            self.population.crossover_individuals(self.crossover_probability)

            # Apply mutation on the offspring
            log_time(start_time, "before mutating")
            self.population.mutate_individuals(self.mutation_probability)

            # Evaluate the individuals with an invalid fitness
            log_time(start_time, "before evaluating")
            await self.population.evaluate_individuals_async()

            log_time(start_time, "before updating the statistics")
            statistics.update_time_series(self.population, log_statistics=True)

            # Check if the best fitness is above the early stopping threshold
            # or if the maximum number of fitness evaluations is reached
            if (
                self.early_stopping
                and statistics.best_fitness_values[-1] >= self.early_stopping_value
            ) or self.num_fitness_evaluations >= self.max_num_fitness_evaluations:
                statistics.fill_time_series(generation, self.num_generations, self.population)
                logging.info("Early stopping!")
                break

            # Check if the probabilities of mutation and crossover are too high
            if self.probability_decay and should_decrease_probability(
                statistics.best_fitness_values[-1], difference_threshold=0.5
            ):
                logging.info("Decreasing mutation and crossover probabilities")
                self.mutate.multiply_probabilities(factor=0.5, log=True)
                self.crossover.multiply_probabilities(factor=0.5, log=True)

        end_time = time.time()
        logging.info(f"Number of evaluations = {self.fitness.num_evaluations}")
        logging.info(f"Total time: {end_time - start_time} seconds")
        statistics.set_scalars(
            runtime=end_time - start_time, num_evaluations=self.fitness.num_evaluations
        )

        # Log the best individual and its fitness
        best_individual = self.fitness.best_individual(statistics.best_individuals)
        best_fitness = best_individual.fitness
        logging.info(
            f"FINAL best individual: {best_individual.fen()}, FINAL best fitness: {best_fitness}"
        )

        # Log the histories of the mutation- and crossover probability distributions
        self.mutate.print_mutation_probability_history()
        self.crossover.print_crossover_probability_history()

        return statistics

    async def cleanup(self) -> None:
        """Cleanup after evolutionary algorithm by closing the multiprocessing pool, saving the
        fitness cache, updating the number of fitness evaluations used by the
        evolutionary algorithm.
        """
        # Cancel all running subprocesses which the fitness evaluator spawned
        self.fitness.cancel_tasks()

        self.pool.close()

        # Add the number of fitness evaluations to the total number of fitness evaluations
        self._num_fitness_evaluations += len(self.fitness.cache) - self._num_fitness_evaluations

        # Save the fitness cache
        self.fitness_cache = self.fitness.cache

        self.fitness = None
        del self.mutate
        del self.crossover
        del self.select
        del self.population


async def main(
    experiment_config_dict: Dict[str, Any], logger: logging.Logger
) -> List[SimpleStatistics]:
    """Repeatedly run the evolutionary algorithm until the maximum number of fitness evaluations
    is reached.

    Args:
        experiment_config_dict (Dict[str, Any]): The experiment config.
        logger (logging.Logger): The logger.

    Returns:
        List[SimpleStatistics]: The statistics of the evolutionary algorithm.
    """

    # Store the start time
    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Extract path to evolutionary algorithm config file
    evolutionary_algorithm_config_name = experiment_config_dict[
        "evolutionary_algorithm_config_name"
    ]

    # Build evolutionary algorithm config
    evolutionary_algorithm_config = SimpleEvolutionaryAlgorithmConfig.from_yaml_file(
        EVOLUTIONARY_ALGORITHM_CONFIG_FOLDER / evolutionary_algorithm_config_name
    )

    # Log the config
    logger.info(f"\nEvolutionary algorithm config:\n{evolutionary_algorithm_config.__dict__}")

    # Store all fens together with their evaluated fitness values in a result file
    result_file_path = RESULT_DIR / f"oracle_queries_{start_time}.txt"

    # Create the result folder if it doesn't exist
    result_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_file_path, "w") as result_file:
        # Store the general experiment config
        for key, value in experiment_config_dict.items():
            result_file.write(f"{key} = {value}\n")

        # Store the evolutionary algorithm config
        for key, value in evolutionary_algorithm_config.__dict__.items():
            result_file.write(f"{key} = {value}\n")

        result_file.write("\n")

    # Create the evolutionary algorithm
    evolutionary_algorithm = DistributedOracleQueryEvolutionaryAlgorithm(
        evolutionary_algorithm_config=evolutionary_algorithm_config,
        experiment_config=experiment_config_dict,
        logger=logger,
        result_file_path=result_file_path,
    )

    # Extract the fitness cache and store the cached values in a file

    run_statistics = []
    start_seed = experiment_config_dict["seed"]
    run_id = 0

    # Run algorithm as long as the number of oracle calls hasn't reached the maximum
    while (
        evolutionary_algorithm.num_fitness_evaluations
        < evolutionary_algorithm.max_num_fitness_evaluations
    ):
        logger.info(f"\n\nStarting run {run_id + 1}")

        # Print the number of evaluation calls so far
        logger.info(
            f"Number of evaluation calls so far: {evolutionary_algorithm.num_fitness_evaluations}"
        )

        # Initialize evolutionary algorithm object
        await evolutionary_algorithm.initialize(start_seed + run_id)

        # Run evolutionary algorithm
        run_statistics.append(await evolutionary_algorithm.run())

        # Cleanup evolutionary algorithm object
        await evolutionary_algorithm.cleanup()

        run_id += 1

    return run_statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # fmt: off
    # Engine parameters
    parser.add_argument("--seed",                type=int,  default=42)  # noqa
    # parser.add_argument("--evolutionary_algorithm_config_name", type=str,  default="config_simple_population.yaml")  # noqa
    parser.add_argument("--evolutionary_algorithm_config_name", type=str,  default="config_simple_population_max_oracle_distributed.yaml")  # noqa
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

    # Parse the arguments
    args = parser.parse_args()

    # Get the experiment config as a dictionary
    experiment_config_dict = get_experiment_params_dict(namespace=args, source_file_path=__file__)

    np.random.seed(args.seed)

    # Build the configs for the fitness function
    experiment_config_dict["fitness_config"] = {}

    # Log the experiment config
    experiment_config_str = "{\n"
    for key, value in experiment_config_dict.items():
        experiment_config_str += f"    {key}: {value},\n"
    experiment_config_str += "\n}"

    logger.info(f"\nExperiment config:\n{experiment_config_str}")

    # Run the evolutionary algorithm 'num_runs_per_config' times
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    statistics: List[SimpleStatistics] = asyncio.run(main(experiment_config_dict, logger))

    # Average the fitness values and unique individual fractions over all runs
    # and compute the standard deviation
    averaged_statistics = SimpleStatistics.average_statistics(statistics)

    # Log the results
    averaged_statistics.log_statistics(use_wandb=not DEBUG)
