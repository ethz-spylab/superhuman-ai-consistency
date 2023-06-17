import logging
import time
from enum import Enum
from operator import attrgetter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chess
import numpy as np

from rl_testing.evolutionary_algorithms.individuals import Individual
from rl_testing.util.util import get_random_state


class SelectionName(Enum):
    SELECT_TOURNAMENT = 0
    SELECT_TOURNAMENT_FAST = 1


def select_tournament_fast(
    individuals: List[Individual],
    rounds: int,
    tournament_fraction: float,
    is_bigger_better: bool = True,
    batch_size: int = 1000,
    _random_state: Optional[np.random.Generator] = None,
) -> List[Individual]:
    """
    This is an optimized version of the simple implementation found in the deap library:
    https://github.com/DEAP/deap/blob/master/deap/tools/selection.py

    Select the best individual among "tournament_size" randomly chosen
    individuals, "rounds" times. The list returned contains
    references to the input "individuals".

    Args:
        individuals (List[Individual]): A list of individuals to select from.
        rounds (int): The number of times to select an individual.
        tournament_fraction (float): The fraction of individuals to select from.
        is_bigger_better (bool, optional): Whether a higher fitness value is better. Defaults to True.
        batch_size (int, optional): The number of individuals to select at a time. Defaults to 1000.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[Individual]: The selected individuals.
    """
    random_state = get_random_state(_random_state)

    assert 0 < tournament_fraction <= 1, "The tournament fraction must be between 0 and 1."
    tournament_size = np.round(tournament_fraction * len(individuals)).astype(int)

    # If the batch size is larger than 1000, warn the user
    if batch_size > 1000:
        logging.warning(
            f"Batch size is set to {batch_size}, which is larger than 1000. This may take a long time and consume a lot of memory."
        )

    # First sort the individuals by fitness.
    individuals.sort(key=attrgetter("fitness"), reverse=is_bigger_better)

    # Compute the individual batch sizes
    batch_sizes = [batch_size] * (rounds // batch_size) + [rounds % batch_size]

    # Select the individuals
    # This does the same like the function 'select_tournament' but is much faster.
    tournament_winner_indices = []
    for current_batch_size in batch_sizes:
        # Compute a 2d array of random numbers
        random_numbers = random_state.random((current_batch_size, len(individuals)))

        # For each row, find the indices of the 'tournament_size' largest random numbers
        tournament_indices = np.argpartition(random_numbers, tournament_size, axis=1)[
            :, -tournament_size:
        ]

        # For each row, find the smallest number and append it to the list of tournament winners
        tournament_winner_indices.extend(np.min(tournament_indices, axis=1))

    # Return the tournament winners
    return [individuals[i] for i in tournament_winner_indices]


def select_tournament(
    individuals: List[Individual],
    rounds: int,
    tournament_size: int,
    find_best_individual: Callable[[List[Individual]], Tuple[Individual, float]],
    _random_state: Optional[np.random.Generator] = None,
) -> List[Individual]:
    """
    Adapted from the deap library: https://github.com/DEAP/deap/blob/master/deap/tools/selection.py
    Select the best individual among "tournament_size" randomly chosen
    individuals, "rounds" times. The list returned contains
    references to the input "individuals".

    Args:
        individuals (List[Individual]): A list of individuals to select from.
        rounds (int): The number of times to select an individual.
        tournament_size (int): The number of individuals to choose from.
        find_best_individual (Callable[[List[Individual]], Tuple[Individual, float]]): A function that finds the best
            individual in a list of individuals.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[Individual]: The selected individuals.

    """
    random_state = get_random_state(_random_state)

    chosen = []
    start_time = time.time()
    for i in range(rounds):
        aspirants = random_state.choice(individuals, tournament_size, replace=False)
        chosen.append(find_best_individual(aspirants)[0])
    return chosen


def _get_linear_candidates(
    grid: List[List[Individual]], row: int, column: int, arm_length: int
) -> List[Individual]:
    """Returns the candidates for the linear selection. A linear selection with arm_length 3 in a 2d grid
    looks like this (x = individuals selected, 0 = individuals not selected):
            0 0 0 x 0 0 0
            0 0 0 x 0 0 0
            0 0 0 x 0 0 0
            x x x x x x x
            0 0 0 x 0 0 0
            0 0 0 x 0 0 0
            0 0 0 x 0 0 0

    Args:
        grid (List[List[Individual]]): The grid of individuals.
        row (int): The row of the current individual.
        column (int): The column of the current individual.
        arm_length (int): The arm length of the linear selection.

    Returns:
        List[Individual]: The candidates for the linear selection.
    """
    num_rows, num_columns = len(grid), len(grid[0])
    # Sanity check on the input parameters
    assert 0 < arm_length, "The arm length must be greater than 0."
    assert 0 <= row < len(grid), "The row must be between 0 and the number of rows."
    assert 0 <= column < len(grid[0]), "The column must be between 0 and the number of columns."

    candidates = [grid[row][column]]
    for offset in range(-arm_length, arm_length + 1):
        if offset != 0:
            if abs(offset) < (num_columns - 1) // 2:
                candidates.append(grid[row][(column + offset) % num_columns])
            if abs(offset) < (num_rows - 1) // 2:
                candidates.append(grid[(row + offset) % num_rows][column])

    return candidates


def _get_diamond_candidates(
    grid: List[List[Individual]], row: int, column: int, l1_threshold: int
) -> List[Individual]:
    """Returns the candidates for the diamond selection. A diamond selection with l1_threshold 3 in a 2d grid
    looks like this (x = individuals selected, 0 = individuals not selected):
            0 0 0 x 0 0 0
            0 0 x x x 0 0
            0 x x x x x 0
            x x x x x x x
            0 x x x x x 0
            0 0 x x x 0 0
            0 0 0 x 0 0 0

    Args:
        grid (List[List[Individual]]): The grid of individuals.
        row (int): The row of the current individual.
        column (int): The column of the current individual.
        l1_threshold (int): The l1 threshold of the diamond selection.
    Returns:
        List[Individual]: The candidates for the diamond selection.
    """
    num_rows, num_columns = len(grid), len(grid[0])
    # Sanity check on the input parameters
    assert 0 < l1_threshold, "The l1 threshold must be greater than 0."
    assert 0 <= row < num_rows, "The row must be between 0 and the number of rows."
    assert 0 <= column < num_columns, "The column must be between 0 and the number of columns."

    candidates = []
    for x_offset in range(-l1_threshold, l1_threshold + 1):
        for y_offset in range(-l1_threshold, l1_threshold + 1):
            if abs(x_offset) + abs(y_offset) <= l1_threshold:
                if abs(x_offset) < (num_rows - 1) // 2 and abs(y_offset) < (num_columns - 1) // 2:
                    candidates.append(
                        grid[(row + x_offset) % num_rows][(column + y_offset) % num_columns]
                    )

    return candidates


def _get_compact_candidates(
    grid: List[List[Individual]], row: int, column: int, l_infinity_threshold: int
) -> List[Individual]:
    """Returns the candidates for the compact selection. A compact selection with l_infinity_threshold 2 in a 2d grid
    looks like this (x = individuals selected, 0 = individuals not selected):
            0 0 0 0 0 0 0
            0 x x x x x 0
            0 x x x x x 0
            0 x x x x x 0
            0 x x x x x 0
            0 x x x x x 0
            0 0 0 0 0 0 0

    Args:
        grid (List[List[Individual]]): The grid of individuals.
        row (int): The row of the current individual.
        column (int): The column of the current individual.
        l_infinity_threshold (int): The l_infinity threshold of the compact selection.
    Returns:
        List[Individual]: The candidates for the compact selection.
    """
    num_rows, num_columns = len(grid), len(grid[0])

    # Sanity check on the input parameters
    assert 0 < l_infinity_threshold, "The l_infinity threshold must be greater than 0."
    assert 0 <= row < num_rows, "The row must be between 0 and the number of rows."
    assert 0 <= column < num_columns, "The column must be between 0 and the number of columns."

    candidates = []
    for x_offset in range(-l_infinity_threshold, l_infinity_threshold + 1):
        for y_offset in range(-l_infinity_threshold, l_infinity_threshold + 1):
            if abs(x_offset) < (num_rows - 1) // 2 and abs(y_offset) < (num_columns - 1) // 2:
                candidates.append(
                    grid[(row + x_offset) % num_rows][(column + y_offset) % num_columns]
                )

    return candidates


def _select_roulette_wheel(
    population: List[Individual],
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> Individual:
    """Selects one individual from the population using roulette wheel selection.

    Args:
        population (List[Individual]): The population of individuals.
        is_bigger_better (bool, optional): Whether a higher fitness is better. Defaults to True.
        random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Individual: The selected individual.
    """
    random_state = get_random_state(_random_state)
    fitnesses = np.array([individual.fitness for individual in population])

    if not is_bigger_better:
        fitnesses = -fitnesses

    fitnesses = fitnesses - fitnesses.min() + 1e-6

    probabilities = fitnesses / fitnesses.sum()
    return random_state.choice(population, p=probabilities)


def _select_binary_tournament(
    population: List[Individual],
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> Individual:
    """Selects one individual from the population using binary tournament selection, meaning
    that two individuals are selected at random and the one with the highest fitness is returned.

    Args:
        population (List[Individual]): The population of individuals.
        is_bigger_better (bool, optional): Whether a higher fitness is better. Defaults to True.
        random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Individual: The selected individual.
    """
    random_state = get_random_state(_random_state)
    if is_bigger_better:
        return max(random_state.choice(population, 2, replace=False), key=lambda x: x.fitness)

    return min(random_state.choice(population, 2, replace=False), key=lambda x: x.fitness)


def _select_rank(
    population: List[Individual],
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
):
    """Selects one individual from the population using rank selection.

    Args:
        population (List[Individual]): The population of individuals.
        is_bigger_better (bool, optional): Whether a higher fitness is better. Defaults to True.
        random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Individual: The selected individual.
    """
    random_state = get_random_state(_random_state)

    if is_bigger_better:
        population_sorted = sorted(population, key=lambda x: -x.fitness)
    else:
        population_sorted = sorted(population, key=lambda x: x.fitness)

    weights = list(range(len(population), 0, -1))
    weights_sum = sum(weights)
    probabilities = [weight / weights_sum for weight in weights]

    return random_state.choice(population_sorted, p=probabilities)


def _apply_select_and_get_function_on_grid(
    grid: List[List[Individual]],
    get_function: Callable[[], Any],
    get_kwargs: Dict[str, Any],
    select_function: Callable[[List[Individual]], Individual],
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> List[List[Individual]]:
    """
    1. Takes a grid of individuals
    2. Applies the get_function to each entry in the grid. This yields a list of individuals for each entry in the grid.
    3. For each individual in the grid, selects one individual from the resulting list of individuals using the select_function.

    Args:
        grid (List[List[Individual]]): A grid of individuals.
        get_function (Callable[[], Any]): The function to apply to each individual in the grid.
        get_kwargs (Dict[str, Any]): The keyword arguments to pass to the get_function.
        select_function (Callable[[List[Individual]], Individual]): The selection function to use.
        is_bigger_better (bool, optional): Whether a higher fitness is better. Defaults to True.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[List[Individual]]: The new grid of individuals.
    """
    random_state = get_random_state(_random_state)
    result_grid = [
        [
            select_function(
                get_function(grid, row, column, **get_kwargs),
                is_bigger_better=is_bigger_better,
                _random_state=random_state,
            )
            for column in range(len(grid[0]))
        ]
        for row in range(len(grid))
    ]
    return result_grid


def select_grid_linear_binary_tournament(
    individuals: List[List[Individual]],
    arm_length: int,
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> List[List[Individual]]:
    """Perform binary tournament selection, using linear candidates from a grid-like population.

    Args:
        individuals (List[List[Individual]]): The input grid
        arm_length (int): The length of the arms of the linear selection
        is_bigger_better (bool, optional): Whether a bigger fitness value is better.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[List[Individual]]: The selected individuals in grid form.
    """
    return _apply_select_and_get_function_on_grid(
        individuals,
        _get_linear_candidates,
        {"arm_length": arm_length},
        _select_binary_tournament,
        is_bigger_better=is_bigger_better,
        _random_state=_random_state,
    )


def select_grid_linear_roulette_wheel(
    individuals: List[List[Individual]],
    arm_length: int,
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
):
    """Perform roulette wheel selection, using linear candidates from a grid-like population.

    Args:
        individuals (List[List[Individual]]): The input grid
        arm_length (int): The length of the arms of the linear selection
        is_bigger_better (bool, optional): Whether a bigger fitness value is better.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[List[Individual]]: The selected individuals in grid form.
    """
    return _apply_select_and_get_function_on_grid(
        individuals,
        _get_linear_candidates,
        {"arm_length": arm_length},
        _select_roulette_wheel,
        is_bigger_better=is_bigger_better,
        _random_state=_random_state,
    )


def select_grid_linear_rank(
    individuals: List[List[Individual]],
    arm_length: int,
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
):
    """Perform rank selection, using linear candidates from a grid-like population.

    Args:
        individuals (List[List[Individual]]): The input grid
        arm_length (int): The length of the arms of the linear selection
        is_bigger_better (bool, optional): Whether a bigger fitness value is better.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[List[Individual]]: The selected individuals in grid form.
    """
    return _apply_select_and_get_function_on_grid(
        individuals,
        _get_linear_candidates,
        {"arm_length": arm_length},
        _select_rank,
        is_bigger_better=is_bigger_better,
        _random_state=_random_state,
    )


def select_grid_diamond_binary_tournament(
    individuals: List[List[Individual]],
    l1_threshold: int,
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> List[List[Individual]]:
    """Perform binary tournament selection, using diamond candidates from a grid-like population.

    Args:
        individuals (List[List[Individual]]): The input grid
        l1_threshold (int): The threshold for the diamond selection
        is_bigger_better (bool, optional): Whether a bigger fitness value is better.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[List[Individual]]: The selected individuals in grid form.
    """
    return _apply_select_and_get_function_on_grid(
        individuals,
        _get_diamond_candidates,
        {"l1_threshold": l1_threshold},
        _select_binary_tournament,
        is_bigger_better=is_bigger_better,
        _random_state=_random_state,
    )


def select_grid_diamond_roulette_wheel(
    individuals: List[List[Individual]],
    l1_threshold: int,
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> List[List[Individual]]:
    """Perform roulette wheel selection, using diamond candidates from a grid-like population.

    Args:
        individuals (List[List[Individual]]): The input grid
        l1_threshold (int): The threshold for the diamond selection
        is_bigger_better (bool, optional): Whether a bigger fitness value is better.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[List[Individual]]: The selected individuals in grid form.
    """
    return _apply_select_and_get_function_on_grid(
        individuals,
        _get_diamond_candidates,
        {"l1_threshold": l1_threshold},
        _select_roulette_wheel,
        is_bigger_better=is_bigger_better,
        _random_state=_random_state,
    )


def select_grid_diamond_rank(
    individuals: List[List[Individual]],
    l1_threshold: int,
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> List[List[Individual]]:
    """Perform rank selection, using diamond candidates from a grid-like population.

    Args:
        individuals (List[List[Individual]]): The input grid
        l1_threshold (int): The threshold for the diamond selection
        is_bigger_better (bool, optional): Whether a bigger fitness value is better.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[List[Individual]]: The selected individuals in grid form.
    """
    return _apply_select_and_get_function_on_grid(
        individuals,
        _get_diamond_candidates,
        {"l1_threshold": l1_threshold},
        _select_rank,
        is_bigger_better=is_bigger_better,
        _random_state=_random_state,
    )


def select_grid_compact_binary_tournament(
    individuals: List[List[Individual]],
    l_infinity_threshold: int,
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> List[List[Individual]]:
    """Perform binary tournament selection, using compact candidates from a grid-like population.

    Args:
        individuals (List[List[Individual]]): The input grid
        l_infinity_threshold (int): The threshold for the compact selection
        is_bigger_better (bool, optional): Whether a bigger fitness value is better.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[List[Individual]]: The selected individuals in grid form.
    """
    return _apply_select_and_get_function_on_grid(
        individuals,
        _get_compact_candidates,
        {"l_infinity_threshold": l_infinity_threshold},
        _select_binary_tournament,
        is_bigger_better=is_bigger_better,
        _random_state=_random_state,
    )


def select_grid_compact_roulette_wheel(
    individuals: List[List[Individual]],
    l_infinity_threshold: int,
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> List[List[Individual]]:
    """Perform roulette wheel selection, using compact candidates from a grid-like population.

    Args:
        individuals (List[List[Individual]]): The input grid
        l_infinity_threshold (int): The threshold for the compact selection
        is_bigger_better (bool, optional): Whether a bigger fitness value is better.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[List[Individual]]: The selected individuals in grid form.
    """
    return _apply_select_and_get_function_on_grid(
        individuals,
        _get_compact_candidates,
        {"l_infinity_threshold": l_infinity_threshold},
        _select_roulette_wheel,
        is_bigger_better=is_bigger_better,
        _random_state=_random_state,
    )


def select_grid_compact_rank(
    individuals: List[List[Individual]],
    l_infinity_threshold: int,
    is_bigger_better: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> List[List[Individual]]:
    """Perform rank selection, using compact candidates from a grid-like population.

    Args:
        individuals (List[List[Individual]]): The input grid
        l_infinity_threshold (int): The threshold for the compact selection
        is_bigger_better (bool, optional): Whether a bigger fitness value is better.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        List[List[Individual]]: The selected individuals in grid form.
    """
    return _apply_select_and_get_function_on_grid(
        individuals,
        _get_compact_candidates,
        {"l_infinity_threshold": l_infinity_threshold},
        _select_rank,
        is_bigger_better=is_bigger_better,
        _random_state=_random_state,
    )


class SelectionFunction:
    def __init__(
        self,
        function: Callable[[chess.Board, Any], chess.Board],
        probability: float = 1.0,
        _random_state: Optional[np.random.Generator] = None,
        *args,
        **kwargs,
    ):
        """A class that contains a selection function and its arguments.

        Args:
            function (Callable[[chess.Board, Any], chess.Board]): The selection function.
            probability (float, optional): The probability of applying the selection function. Defaults to 1.0.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.
            *args: The arguments to pass to the selection function.
            **kwargs: The keyword arguments to pass to the selection function.
        """
        self.function = function
        self.probability = probability
        self.args = args
        self.kwargs = kwargs
        self.random_state = get_random_state(_random_state)

    def __id__(self) -> int:
        return f"SelectionFunction({self.function.__name__})"

    def __hash__(self) -> int:
        return hash(self.__id__())

    def __eq__(self, other: Any) -> bool:
        return self.__id__() == other.__id__()

    def __call__(
        self, population: List[Individual], *args: Any, **kwargs: Any
    ) -> List[Individual]:
        """Call the selection function and return the selected individuals.

        Args:
            population (List[Individual]): The population to select from.

        Returns:
            List[Individual]: The selected individuals.
        """
        return self.function(population, *self.args, *args, **self.kwargs, **kwargs)


class Selector:
    def __init__(
        self,
        selection_strategy: str = "all",
        num_selection_functions: Optional[int] = None,
        _random_state: Optional[np.random.Generator] = None,
    ):
        """A class that selects individuals from a population.

        Args:
            selection_strategy (str, optional): The selection strategy to use. Must be one of
                ["all", "one_random", "n_random"] where "all" applies all selection functions, "one_random" applies one
                random selection function, and "n_random" applies "num_selection_functions" random selection functions.
                Defaults to "all".
            num_selection_functions (Optional[int], optional): The number of selection functions to apply if the selection
                strategy is "n_random". Defaults to None.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.
        """
        self.selection_functions: List[SelectionFunction] = []
        self.selection_strategy = selection_strategy
        self.num_selection_functions = num_selection_functions
        self.random_state = get_random_state(_random_state)

        assert self.selection_strategy in [
            "all",
            "one_random",
            "n_random",
        ], f"Selection strategy must be one of ['all', 'one_random', 'n_random'], got {self.selection_strategy}."

        if self.selection_strategy == "n_random":
            assert (
                self.num_selection_functions is not None
            ), "Number of selections must be specified if selection strategy is 'n_random'."

    def register_selection_function(
        self,
        functions: Union[
            Callable[[List[Individual], Any], List[Individual]],
            List[Callable[[List[Individual], Any], List[Individual]]],
        ],
        probability: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register a selection function.

        Args:
            selection_function (function): The selection function to register.
        """
        if not isinstance(functions, list):
            functions = [functions]

        for function in functions:
            self.selection_functions.append(
                SelectionFunction(function, probability, self.random_state, *args, **kwargs)
            )

    def __call__(
        self, individuals: List[Individual], *new_args: Any, **new_kwargs: Any
    ) -> List[Individual]:
        """Select individuals from a population according to the selection strategy.

        Args:
            individuals (List[Individual]): The population to select from.

        Raises:
            ValueError: If the selection strategy is invalid.

        Returns:
            List[Individual]: The selected individuals.
        """
        selection_functions_to_apply: List[SelectionFunction] = []

        if self.selection_strategy == "all":
            selection_functions_to_apply = self.selection_functions
        elif self.selection_strategy == "one_random":
            selection_functions_to_apply = [self.random_state.choice(self.selection_functions)]
        elif self.selection_strategy == "n_random":
            selection_functions_to_apply = self.random_state.choice(
                self.selection_functions, self.num_selection_functions, replace=False
            )
        else:
            raise ValueError(f"Invalid selection strategy {self.selection_strategy}.")

        selected_individuals = []
        for selection_function in selection_functions_to_apply:
            selected_individuals.extend(
                selection_function(
                    individuals,
                    _random_state=self.random_state,
                    *new_args,
                    **new_kwargs,
                )
            )

        return selected_individuals


if __name__ == "__main__":
    import time

    from rl_testing.evolutionary_algorithms.fitnesses import EditDistanceFitness
    from rl_testing.evolutionary_algorithms.individuals import BoardIndividual

    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    population_size = 14000
    tournament_fraction = 0.6969565124339143

    # Create a list of fake individuals
    individuals = [BoardIndividual() for _ in range(population_size)]

    # Assign random fitness values to the individuals
    for individual in individuals:
        individual.fitness = np.random.random()

    # Create two selectors
    selector_simple = Selector(selection_strategy="all")
    selector_fast = Selector(selection_strategy="all")

    # Register the selection functions
    fitness = EditDistanceFitness("3r3k/7p/2p1np2/4p1p1/1Pq1P3/2Q2P2/P4RNP/2R4K b - - 0 42")
    selector_simple.register_selection_function(
        select_tournament,
        rounds=population_size,
        tournament_size=int(population_size * tournament_fraction),
        find_best_individual=fitness.best_individual,
    )
    selector_fast.register_selection_function(
        select_tournament_fast,
        rounds=population_size,
        tournament_size=int(population_size * tournament_fraction),
        is_bigger_better=False,
    )

    # Select individuals
    start_time = time.time()
    selected_individuals_simple = selector_simple(individuals)
    logging.info(f"Time simple selection: {round(time.time() - start_time,3)} seconds.")
    start_time = time.time()
    selected_individuals_fast = selector_fast(individuals)
    logging.info(f"Time fast selection: {round(time.time() - start_time,3)} seconds.")

    # Print the average fitness of the selected individuals
    logging.info(
        f"Average fitness simple: {np.mean([individual.fitness for individual in selected_individuals_simple])}"
    )
    logging.info(
        f"Average fitness fast: {np.mean([individual.fitness for individual in selected_individuals_fast])}"
    )
