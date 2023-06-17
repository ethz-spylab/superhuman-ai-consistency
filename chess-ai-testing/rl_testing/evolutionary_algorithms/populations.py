import abc
from itertools import chain
from multiprocessing.pool import Pool
from typing import List, Optional, Union

import numpy as np
from rl_testing.evolutionary_algorithms.crossovers import Crossover
from rl_testing.evolutionary_algorithms.fitnesses import Fitness
from rl_testing.evolutionary_algorithms.individuals import Individual
from rl_testing.evolutionary_algorithms.mutations import Mutator
from rl_testing.evolutionary_algorithms.selections import Selector
from rl_testing.util.util import FakePool, get_random_state


class Population(abc.ABC):
    def __init__(
        self,
        fitness: Fitness,
        mutator: Mutator,
        crossover: Crossover,
        pool: Optional[Pool] = None,
        _random_state: Optional[int] = None,
    ):
        self.fitness: Fitness = fitness
        self.mutator: Mutator = mutator
        self.crossover: Crossover = crossover
        self.pool = pool if pool is not None else FakePool()
        self.selector: Selector  # Needs to be set by subclass
        self.num_processes = self.pool._processes
        self.random_state = get_random_state(_random_state)

    @property
    @abc.abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def flat_population(self) -> List[Individual]:
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_individuals(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def evaluate_individuals_async(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def create_next_generation(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def crossover_individuals(self, crossover_prob: float) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_individuals(self, mutation_prob: float) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def best_individual(self) -> Individual:
        raise NotImplementedError

    @abc.abstractmethod
    def worst_individual(self) -> Individual:
        raise NotImplementedError

    @abc.abstractmethod
    def average_fitness(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def unique_individual_fraction(self) -> float:
        raise NotImplementedError


class SimplePopulation(Population):
    def __init__(
        self,
        individuals: List[Individual],
        fitness: Fitness,
        mutator: Mutator,
        crossover: Crossover,
        selector: Selector,
        pool: Optional[Pool] = None,
        _random_state: Optional[int] = None,
    ):
        super().__init__(fitness, mutator, crossover, pool, _random_state)
        self.individuals = individuals
        self.selector = selector

    @property
    def size(self) -> int:
        return len(self.individuals)

    @property
    def flat_population(self) -> List[Individual]:
        return self.individuals

    def initialize(self) -> None:
        return

    def evaluate_individuals(self) -> None:
        for individual in self.individuals:
            if individual.fitness is None:
                individual.fitness = self.fitness.evaluate(individual)

    async def evaluate_individuals_async(self) -> None:
        unevaluated_individuals: List[Individual] = []
        for individual in self.individuals:
            if individual.fitness is None:
                unevaluated_individuals.append(individual)

        for individual, fitness_val in zip(
            unevaluated_individuals, await self.fitness.evaluate_async(unevaluated_individuals)
        ):
            individual.fitness = fitness_val

    def create_next_generation(self) -> None:
        population_size = len(self.individuals)
        num_workers = self.num_processes

        # prepare the chunk sizes for the selection
        chunk_sizes = [population_size // num_workers] * num_workers + [
            population_size % num_workers
        ]

        # Select the individuals in parallel
        offspring: List[Individual] = []
        results_async = [
            self.pool.apply_async(
                self.selector,
                # args=(population, chunk_size_i),
                kwds={"individuals": self.individuals, "rounds": chunk_size_i},
            )
            for chunk_size_i in chunk_sizes
        ]
        for result_async in results_async:
            offspring += result_async.get()

        self.individuals = [individual.copy() for individual in offspring]

    def _crossover_individuals(
        self, individuals: List[Individual], crossover_prob: float
    ) -> List[Individual]:
        mated_children: List[Individual] = []

        # Select potential mating candidates
        self.random_state.shuffle(individuals)
        couple_candidates = list(zip(individuals[::2], individuals[1::2]))

        # Filter out the individuals that will mate
        random_values = self.random_state.random(size=len(individuals) // 2)
        mating_candidates = [
            couple_candidates[i]
            for i, random_value in enumerate(random_values)
            if random_value < crossover_prob
        ]
        single_children = [
            couple_candidates[i]
            for i, random_value in enumerate(random_values)
            if random_value >= crossover_prob
        ]

        # Apply crossover on the mating candidates
        random_seeds = self.random_state.integers(0, 2**63, len(mating_candidates))
        for individual1, individual2 in chain(
            single_children,
            self.pool.starmap(self.crossover, zip(mating_candidates, random_seeds)),
        ):
            mated_children.append(individual1)
            mated_children.append(individual2)

        return mated_children

    def crossover_individuals(self, crossover_prob: float) -> None:
        mated_children = self._crossover_individuals(self.individuals, crossover_prob)
        self.individuals = mated_children

    def _mutate_individuals(
        self, individuals: List[Individual], mutation_prob: float
    ) -> List[Individual]:
        mutated_children: List[Individual] = []
        random_values = self.random_state.random(size=len(individuals))

        # Filter out the individuals that will mutate
        mutation_candidates = [
            individuals[i]
            for i, random_value in enumerate(random_values)
            if random_value < mutation_prob
        ]
        non_mutation_candidates = [
            individuals[i]
            for i, random_value in enumerate(random_values)
            if random_value >= mutation_prob
        ]

        # Apply mutation on the mutation candidates
        random_seeds = self.random_state.integers(0, 2**63, len(mutation_candidates))
        mutated_children: List[Individual] = []
        for individual in chain(
            self.pool.starmap(self.mutator, zip(mutation_candidates, random_seeds)),
            non_mutation_candidates,
        ):
            mutated_children.append(individual)

        return mutated_children

    def mutate_individuals(self, mutation_prob: float) -> None:
        mutated_children = self._mutate_individuals(self.individuals, mutation_prob)
        self.individuals = mutated_children

    def best_individual(self) -> Individual:
        assert all(
            individual.fitness is not None for individual in self.individuals
        ), "All individuals must have a fitness value assigned before calling this operation."
        return self.fitness.best_individual(self.individuals)

    def worst_individual(self) -> Individual:
        assert all(
            individual.fitness is not None for individual in self.individuals
        ), "All individuals must have a fitness value assigned before calling this operation."
        return self.fitness.worst_individual(self.individuals)

    def average_fitness(self) -> float:
        assert all(
            individual.fitness is not None for individual in self.individuals
        ), "All individuals must have a fitness value assigned before calling this operation."
        return np.mean([individual.fitness for individual in self.individuals])

    def unique_individual_fraction(self) -> float:
        return (
            len(set(self.individuals)) / len(self.individuals)
            if len(self.individuals) > 0
            else 0.0
        )


class AdaptiveWeightPopulation(SimplePopulation):
    def __init__(
        self,
        individuals: List[Individual],
        fitness: Fitness,
        mutator: Mutator,
        crossover: Crossover,
        selector: Selector,
        pool: Optional[Pool] = None,
        _random_state: Optional[int] = None,
    ):
        super().__init__(
            individuals,
            fitness,
            mutator,
            crossover,
            selector,
            pool,
            _random_state,
        )
        self._population_size = len(individuals)
        self._mated_individuals: List[Individual] = []
        self._mutated_individuals: List[Individual] = []

    @property
    def size(self) -> int:
        return self._population_size

    def initialize(self) -> None:
        pass

    def create_next_generation(self) -> None:
        num_workers = self.num_processes

        # prepare the chunk sizes for the selection
        chunk_sizes = [self._population_size // num_workers] * num_workers + [
            self._population_size % num_workers
        ]

        # Select the individuals in parallel
        offspring: List[Individual] = []
        results_async = [
            self.pool.apply_async(
                self.selector,
                # args=(population, chunk_size_i),
                kwds={"individuals": self.individuals, "rounds": chunk_size_i},
            )
            for chunk_size_i in chunk_sizes
        ]
        for result_async in results_async:
            offspring += result_async.get()

        self.individuals = [individual.copy() for individual in offspring]

    def crossover_individuals(self, crossover_prob: float) -> None:
        self._mated_individuals = self._crossover_individuals(self.individuals, crossover_prob)

    def mutate_individuals(self, mutation_prob: float) -> None:
        self._mutated_individuals = self._mutate_individuals(self.individuals, mutation_prob)
        self.individuals = self._mutated_individuals + self._mated_individuals


class CellularPopulation(Population):
    CROSSOVER_MODES = ["one_neighbor", "two_neighbors"]

    def __init__(
        self,
        individuals: List[Individual],
        num_rows: int,
        num_columns: int,
        crossover_mode: str,
        fitness: Fitness,
        mutator: Mutator,
        crossover: Crossover,
        selector: Selector,
        pool: Optional[Pool] = None,
        apply_operators_on_original_grid: bool = True,
        _random_state: Optional[int] = None,
    ):
        assert (
            len(individuals) == num_rows * num_columns
        ), "The number of individuals must be equal to the number of rows times the number of columns."

        assert (
            crossover_mode in self.CROSSOVER_MODES
        ), f"Invalid crossover mode: {crossover_mode}. Use one of {self.CROSSOVER_MODES}."

        super().__init__(fitness, mutator, crossover, pool, _random_state)
        self.num_rows = num_rows
        self.num_cols = num_columns
        self.crossover_mode = crossover_mode
        self.selector = selector
        self.apply_operators_on_original_grid = apply_operators_on_original_grid

        # Create a 2D grid of individuals
        self.grid: List[List[Individual]] = [
            individuals[start : start + num_columns]
            for start in range(0, len(individuals), num_columns)
        ]
        self.mutation_grid: List[List[Individual]] = []
        self.crossover_grid: List[List[Individual]] = []

    @property
    def size(self) -> int:
        return self.num_rows * self.num_cols

    @property
    def flat_population(self) -> List[Individual]:
        return [individual for row in self.grid for individual in row]

    def initialize(self):
        pass

    def evaluate_individuals(self) -> None:
        for row in self.grid:
            for individual in row:
                if individual.fitness is None:
                    individual.fitness = self.fitness.evaluate(individual)

    async def evaluate_individuals_async(self) -> None:
        grids = [self.grid, self.mutation_grid, self.crossover_grid]
        unevaluated_individuals: List[Individual] = []
        for grid in grids:
            for row in grid:
                for individual in row:
                    if individual.fitness is None:
                        unevaluated_individuals.append(individual)

        for individual, fitness_val in zip(
            unevaluated_individuals, await self.fitness.evaluate_async(unevaluated_individuals)
        ):
            individual.fitness = fitness_val

    def create_next_generation(self) -> None:
        # For each position in the grid, select the individual with the best fitness from the set of candidate grids
        candidate_grids = [self.grid, self.mutation_grid, self.crossover_grid]
        candidate_grids = [grid for grid in candidate_grids if grid]

        for row in range(self.num_rows):
            for column in range(self.num_cols):
                self.grid[row][column] = self.fitness.best_individual(
                    [grid[row][column] for grid in candidate_grids]
                )

    def crossover_individuals(self, crossover_prob: float) -> None:
        # For each individual, select one neighbor
        first_parents = self.selector(self.grid)

        # Depending on crossover mode, select a second neighbor
        if self.crossover_mode == "one_neighbor":
            second_parents = self.grid
        elif self.crossover_mode == "two_neighbors":
            second_parents = self.selector(self.grid)

        # Create random numbers for each individual
        random_seeds = self.random_state.integers(0, 2**63, (self.num_rows, self.num_cols))

        # Crossover the individuals
        crossover_grid_tuples = [
            [
                self.pool.apply_async(
                    self.crossover,
                    kwds={
                        "individual_tuple": (
                            first_parents[row][column],
                            second_parents[row][column],
                        ),
                        "random_seed": random_seeds[row][column],
                    },
                )
                for column in range(self.num_cols)
            ]
            for row in range(self.num_rows)
        ]

        # Get the results
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                crossover_grid_tuples[row][column] = crossover_grid_tuples[row][column].get()

        # Each entry in the 2d grid now contains a tuple of individuals. Choose for each tuple
        # one individual at random
        self.crossover_grid = [
            [self.random_state.choice(individual_tuple) for individual_tuple in row]
            for row in crossover_grid_tuples
        ]

    def mutate_individuals(self, mutation_prob: float) -> None:
        # Depending on whether to stack the operators or not, create a new grid of individuals
        if self.apply_operators_on_original_grid:
            grid = self.grid
        else:
            grid = self.crossover_grid

        # Create random numbers for each individual
        random_seeds = self.random_state.integers(0, 2**63, (self.num_rows, self.num_cols))

        # Mutate the individuals
        result_grid = [
            [
                self.pool.apply_async(
                    self.mutator,
                    kwds={"individual": individual, "random_seed": random_seed},
                )
                for individual, random_seed in zip(row, random_seeds[row_index])
            ]
            for row_index, row in enumerate(grid)
        ]

        # Get the results
        self.mutation_grid = [[result.get() for result in row] for row in result_grid]

    def best_individual(self) -> Individual:
        flatted_individuals = self.flat_population
        assert all(
            individual.fitness is not None for individual in flatted_individuals
        ), "Cannot determine best individual if not all individuals have a fitness value."
        return self.fitness.best_individual(flatted_individuals)

    def worst_individual(self) -> Individual:
        flatted_individuals = self.flat_population
        assert all(
            individual.fitness is not None for individual in flatted_individuals
        ), "Cannot determine worst individual if not all individuals have a fitness value."
        return self.fitness.worst_individual(flatted_individuals)

    def average_fitness(self) -> float:
        flattened_individuals = self.flat_population
        assert all(
            individual.fitness is not None for individual in flattened_individuals
        ), "All individuals must have a fitness value assigned before calling this operation."
        return np.mean([individual.fitness for individual in flattened_individuals])

    def unique_individual_fraction(self) -> float:
        flattened_individuals = self.flat_population
        return (
            len(set(flattened_individuals)) / self.size if len(flattened_individuals) > 0 else 0.0
        )
