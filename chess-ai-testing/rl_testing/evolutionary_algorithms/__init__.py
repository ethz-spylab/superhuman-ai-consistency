from typing import Any, Callable, List, Optional

import numpy as np

import rl_testing.evolutionary_algorithms.crossovers
import rl_testing.evolutionary_algorithms.mutations
import rl_testing.evolutionary_algorithms.selections
from rl_testing.evolutionary_algorithms.crossovers import Crossover
from rl_testing.evolutionary_algorithms.mutations import Mutator
from rl_testing.evolutionary_algorithms.selections import Selector
from rl_testing.util.util import get_random_state


def is_mutator_initializer(object) -> bool:
    return (
        hasattr(object, "mutator_attributes")
        and hasattr(object, "mutation_functions")
        and hasattr(object, "global_mutate_kwargs")
    )


def is_crossover_initializer(object) -> bool:
    return (
        hasattr(object, "crossover_attributes")
        and hasattr(object, "crossover_functions")
        and hasattr(object, "global_crossover_kwargs")
    )


def is_selector_initializer(object) -> bool:
    return (
        hasattr(object, "selector_attributes")
        and hasattr(object, "selection_functions")
        and hasattr(object, "global_select_kwargs")
    )


def get_initialized_mutator(
    config: Any,
    _random_state: Optional[np.random.Generator] = None,
) -> Mutator:
    """Initialize a mutator object from a config object.

    Args:
        config (Any): A config object that has a mutation_function_info attribute.
        _random_state (Optional[np.random.Generator], optional): A random state. Defaults to None.

    Returns:
        Mutator: An initialized mutator object.
    """

    random_state = get_random_state(_random_state)

    # Make sure that the provided config object has a mutation_functions attribute.
    if not is_mutator_initializer(config):
        raise TypeError(
            "The provided config object does not have all necessary attributes to initialize a mutator."
        )

    # Initialize the mutator object.
    mutator = Mutator(
        **config.mutator_attributes,
        _random_state=random_state,
    )

    # Register the mutation functions.
    for mutation_function_name in config.mutation_functions:
        # Get the mutation function specified by 'mutation_function_name'.
        mutation_function = getattr(
            rl_testing.evolutionary_algorithms.mutations, mutation_function_name
        )

        # Copy the mutation function kwargs and remove the weight.
        mutation_function_kwargs = config.mutation_functions[mutation_function_name].get(
            "kwargs", {}
        )

        # Register the mutation function.
        mutator.register_mutation_function(
            functions=mutation_function,
            probability=config.mutation_functions[mutation_function_name]["weight"],
            **mutation_function_kwargs,
            **config.global_mutate_kwargs,
        )

    return mutator


def get_initialized_crossover(
    config: Any,
    _random_state: Optional[np.random.Generator] = None,
) -> Crossover:
    """Initialize a crossover object from a config object.

    Args:
        config (Any): A config object that has a crossover_function_info attribute.
        _random_state (Optional[np.random.Generator], optional): A random state. Defaults to None.

    Returns:
        Crossover: An initialized crossover object.
    """

    random_state = get_random_state(_random_state)

    # Make sure that the provided config object has a crossover_functions attribute.
    if not is_crossover_initializer(config):
        raise TypeError(
            "The provided config object does not have all necessary attributes to initialize a crossover."
        )

    # Initialize the crossover object.
    crossover = Crossover(
        **config.crossover_attributes,
        _random_state=random_state,
    )

    # Register the crossover functions.
    for crossover_function_name in config.crossover_functions:
        # Get the crossover function specified by 'crossover_function_name'.
        crossover_function = getattr(
            rl_testing.evolutionary_algorithms.crossovers, crossover_function_name
        )

        # Copy the crossover function kwargs and remove the weight.
        crossover_function_kwargs = config.crossover_functions[crossover_function_name].get(
            "kwargs", {}
        )

        # Register the crossover function.
        crossover.register_crossover_function(
            functions=crossover_function,
            probability=config.crossover_functions[crossover_function_name]["weight"],
            **crossover_function_kwargs,
            **config.global_crossover_kwargs,
        )

    return crossover


def get_initialized_selector(
    config: Any,
    _random_state: Optional[np.random.Generator] = None,
) -> Selector:
    """Initialize a selector object from a config object.

    Args:
        config (Any): A config object that has a selector_function_info attribute.
        _random_state (Optional[np.random.Generator], optional): A random state. Defaults to None.

    Returns:
        Selector: An initialized selector object.
    """

    random_state = get_random_state(_random_state)

    # Make sure that the provided config object has a selector_functions attribute.
    if not is_selector_initializer(config):
        raise TypeError(
            "The provided config object does not have all necessary attributes to initialize a selector."
        )

    # Initialize the selector object.
    selector = Selector(
        **config.selector_attributes,
        _random_state=random_state,
    )

    # Register the selector functions.
    for selector_function_name in config.selection_functions:
        # Get the selector function specified by 'selector_function_name'.
        selector_function = getattr(
            rl_testing.evolutionary_algorithms.selections, selector_function_name
        )

        # Copy the selector function kwargs and remove the weight.
        selector_function_kwargs = config.selection_functions[selector_function_name].get(
            "kwargs", {}
        )

        # Register the selector function.
        selector.register_selection_function(
            functions=selector_function,
            probability=config.selection_functions[selector_function_name]["weight"],
            **selector_function_kwargs,
            **config.global_select_kwargs,
        )

    return selector
