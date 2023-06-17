import abc
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import wandb

from rl_testing.evolutionary_algorithms.individuals import Individual
from rl_testing.evolutionary_algorithms.populations import Population

BestFitnessValue = float
WorstFitnessValue = float
AverageFitnessValue = float
UniqueIndividualFraction = float


class Statistics(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "update_time_series")
            and callable(subclass.update_time_series)
            and hasattr(subclass, "fill_time_series")
            and callable(subclass.fill_time_series)
            and hasattr(subclass, "set_scalars")
            and callable(subclass.set_scalars)
            and hasattr(subclass, "log_statistics")
            and callable(subclass.log_statistics)
            and hasattr(subclass, "average_statistics")
            and callable(subclass.average_statistics)
            or NotImplemented
        )

    @abc.abstractmethod
    def update_time_series(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def fill_time_series(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def set_scalars(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_statistics(self, use_wandb: bool = False) -> None:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def average_statistics(statistics: List["Statistics"]) -> "Statistics":
        raise NotImplementedError


class SimpleStatistics(Statistics):
    def __init__(
        self,
    ):
        """Initialize the statistics object."""
        # Initialize the scalars
        self.runtime: Optional[float] = None
        self.num_evaluations: Optional[int] = None

        # Initialize the time series
        self.best_individuals: List[Individual] = []
        self.best_fitness_values: List[BestFitnessValue] = []
        self.average_fitness_values: List[AverageFitnessValue] = []
        self.worst_fitness_values: List[WorstFitnessValue] = []
        self.unique_individual_fractions: List[UniqueIndividualFraction] = []

        # Optional standard deviations in case that the statistics is used to store the average of multiple runs
        self.runtime_std: Optional[float] = None
        self.best_fitness_values_std: Optional[List[float]] = None
        self.average_fitness_values_std: Optional[List[float]] = None
        self.worst_fitness_values_std: Optional[List[float]] = None
        self.unique_individual_fractions_std: Optional[List[float]] = None
        self.stores_averages = False

    def update_time_series(
        self,
        population: Population,
        log_statistics: bool = False,
    ) -> None:
        """Update the statistics of the population.

        Args:
            population (Population): The population to update the statistics of.
            log_statistics (bool, optional): Whether to log the statistics. Defaults to False.
        """
        # Get the statistics of the population
        best_individual = population.best_individual()
        best_fitness = best_individual.fitness
        worst_individual = population.worst_individual()
        worst_fitness = worst_individual.fitness
        average_fitness = population.average_fitness()
        unique_individual_fraction = population.unique_individual_fraction()

        # Log the statistics
        if log_statistics:
            logging.info(f"{best_individual = }, {best_fitness = }")
            logging.info(f"{worst_individual = }, {worst_fitness = }")
            logging.info(f"{average_fitness = }")
            logging.info(
                "Number of unique individuals ="
                f" {np.round(unique_individual_fraction * population.size).astype(int)}"
            )
            logging.info(f"{best_individual.history = }")

        # Update the statistics
        self.best_individuals.append(best_individual.copy())
        self.best_fitness_values.append(best_fitness)
        self.average_fitness_values.append(average_fitness)
        self.worst_fitness_values.append(worst_fitness)
        self.unique_individual_fractions.append(unique_individual_fraction)

    def fill_time_series(
        self,
        generation: int,
        num_generations: int,
        population: Population,
    ) -> None:
        """Fill the time series with the last value.

        Args:
            generation (int): The current generation.
            num_generations (int): The total number of generations.
            population (Population): The population to determine the statistics which are used to fill the time series.
        """
        num_generations_remaining = num_generations - generation - 1

        # Get the statistics of the population
        best_individual = population.best_individual()
        best_fitness = best_individual.fitness
        worst_individual = population.worst_individual()
        worst_fitness = worst_individual.fitness
        average_fitness = population.average_fitness()
        unique_individual_fraction = population.unique_individual_fraction()

        # Fill up the lists with the last value
        self.best_individuals.extend([best_individual.copy()] * num_generations_remaining)
        self.best_fitness_values.extend([best_fitness] * num_generations_remaining)
        self.average_fitness_values.extend([average_fitness] * num_generations_remaining)
        self.worst_fitness_values.extend([worst_fitness] * num_generations_remaining)
        self.unique_individual_fractions.extend(
            [unique_individual_fraction] * num_generations_remaining
        )

    def set_scalars(
        self,
        *,
        runtime: Optional[float] = None,
        num_evaluations: Optional[int] = None,
    ):
        """Set the constants of the statistics.

        Args:
            runtime (Optional[float], optional): The runtime of the algorithm. Defaults to None.
            num_evaluations (Optional[int], optional): The number of evaluations of the algorithm. Defaults to None.
        """
        if runtime is not None:
            self.runtime = runtime
        if num_evaluations is not None:
            self.num_evaluations = num_evaluations

    def log_statistics(self, use_wandb: bool = False) -> None:
        """Log the statistics.

        Args:
            use_wandb (bool, optional): Whether to use Weights & Biases. Defaults to False.
        """
        # Build dictionary with all scalars and time series which aren't None
        scalar_data_dict: Dict[str, Any] = {}
        time_series_data_dict: Dict[str, List[Any]] = {}
        for key, value in self.__dict__.items():
            if value:  # This is only true if value is not None AND not an empty list
                if isinstance(value, list):
                    time_series_data_dict[key] = value
                else:
                    scalar_data_dict[key] = value

        # Make sure that all time series have the same length
        time_series_lengths = [len(time_series) for time_series in time_series_data_dict.values()]
        assert len(set(time_series_lengths)) == 1, "All time series must have the same length."

        num_generations = time_series_lengths[0]

        # Log the statistics
        logging.info(f"Scalars:\n{scalar_data_dict = }")
        logging.info(f"Time series:\n{time_series_data_dict = }")

        # Log the statistics to Weights & Biases
        if use_wandb:
            # Log all scalars
            wandb.log(scalar_data_dict)

            # Log all time series
            for generation in range(num_generations):
                data_dict = {
                    key: time_series[generation]
                    for key, time_series in time_series_data_dict.items()
                }
                wandb.log(data_dict, step=generation)

    @staticmethod
    def average_statistics(statistics: List["SimpleStatistics"]) -> "SimpleStatistics":
        """Average the statistics of multiple runs.

        Args:
            statistics (List[SimpleStatistics]): The statistics of multiple runs.

        Returns:
            SimpleStatistics: The averaged statistics.
        """
        # Get the number of runs
        num_runs = len(statistics)

        # Initialize the averaged statistics
        averaged_statistics = SimpleStatistics()
        averaged_statistics.stores_averages = True

        # Iterate over all scalar parameters which should be averaged
        for parameter_name in ["runtime", "num_evaluations"]:
            # Gather the parameter values of all runs
            parameter_values = [getattr(statistics[i], parameter_name) for i in range(num_runs)]
            parameter_values = [value for value in parameter_values if value is not None]
            if not parameter_values:
                continue

            # Calculate the average and standard deviation of the parameter values
            average = np.mean(parameter_values)
            std = np.std(parameter_values)

            # Set the averaged parameter values
            setattr(averaged_statistics, parameter_name, average)
            setattr(averaged_statistics, f"{parameter_name}_std", std)

        # Iterate over all time series parameters which should be averaged
        for parameter_name in [
            "best_fitness_values",
            "average_fitness_values",
            "worst_fitness_values",
            "unique_individual_fractions",
        ]:
            # Gather the parameter values of all runs
            parameter_values = [getattr(statistics[i], parameter_name) for i in range(num_runs)]

            # Make sure that all time series have the same length
            # Pad the time series with np.nan if necessary
            time_series_lengths = [len(time_series) for time_series in parameter_values]
            max_time_series_length = max(time_series_lengths)
            for i in range(num_runs):
                if time_series_lengths[i] < max_time_series_length:
                    parameter_values[i].extend(
                        [np.nan] * (max_time_series_length - time_series_lengths[i])
                    )

            # Calculate the average and standard deviation of the parameter values
            averages = list(np.nanmean(parameter_values, axis=0))
            stds = list(np.nanstd(parameter_values, axis=0))

            # Set the averaged parameter values
            setattr(averaged_statistics, parameter_name, averages)
            setattr(averaged_statistics, f"{parameter_name}_std", stds)

        return averaged_statistics
