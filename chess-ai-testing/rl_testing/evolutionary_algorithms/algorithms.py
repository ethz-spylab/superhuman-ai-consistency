import abc
import logging
from typing import List

from rl_testing.config_parsers.evolutionary_algorithm_config_parser import (
    EvolutionaryAlgorithmConfig,
)
from rl_testing.evolutionary_algorithms.individuals import Individual
from rl_testing.evolutionary_algorithms.populations import Population
from rl_testing.evolutionary_algorithms.statistics import Statistics


class AsyncEvolutionaryAlgorithm(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "initialize")
            and callable(subclass.initialize_once)
            and hasattr(subclass, "run")
            and callable(subclass.run)
            and hasattr(subclass, "cleanup")
            and callable(subclass.cleanup)
            or NotImplemented
        )

    @abc.abstractmethod
    def __init__(self, config: EvolutionaryAlgorithmConfig) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def initialize(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def run(self) -> None:
        """Run the evolutionary algorithm."""
        raise NotImplementedError

    @abc.abstractmethod
    async def cleanup(self) -> None:
        raise NotImplementedError
