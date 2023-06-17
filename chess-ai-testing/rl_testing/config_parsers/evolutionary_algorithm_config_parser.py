import configparser
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from rl_testing.config_parsers.abstract import Config


class EvolutionaryAlgorithmConfig(Config):
    CONFIG_FOLDER = Path(".")
    DEFAULT_CONFIG_NAME = Path("default.ini")
    REQUIRED_ATTRIBUTES = [
        "config_type",
        "mutator_attributes",
        "mutation_functions",
        "crossover_attributes",
        "crossover_functions",
        "selector_attributes",
        "selection_functions",
    ]
    OPTIONAL_ATTRIBUTES = [
        "global_mutate_kwargs",
        "global_crossover_kwargs",
        "global_select_kwargs",
    ]

    def __init__(
        self,
        config: Union[Dict[str, Dict[str, Any]], configparser.ConfigParser],
        _initialize: bool = True,
    ):
        # Initialize the parameters
        self.config_type = None
        self.global_mutate_kwargs = {}
        self.global_crossover_kwargs = {}
        self.global_select_kwargs = {}

        # Operator attributes
        self.mutator_attributes = {}
        self.mutation_functions = {}
        self.crossover_attributes = {}
        self.crossover_functions = {}
        self.selector_attributes = {}
        self.selection_functions = {}

        # Assign the parameters from the provided config file
        if _initialize:
            self.set_parameters(config=config)
            self.check_parameters()

    def set_parameter(self, section: str, name: str, value: Any) -> None:
        if isinstance(value, str):
            value = self.parse_string(value, raise_error=False)

        if section == "General":
            if (
                name in EvolutionaryAlgorithmConfig.REQUIRED_ATTRIBUTES
                or name in EvolutionaryAlgorithmConfig.OPTIONAL_ATTRIBUTES
            ):
                setattr(self, name, value)
        elif section == "Mutator":
            self.mutator_attributes[name] = value
        elif section == "Crossover":
            self.crossover_attributes[name] = value
        elif section == "Selector":
            self.selector_attributes[name] = value
        elif section == "Operators":
            if "mutate" in name:
                if "global" in name:
                    self.global_mutate_kwargs = value
                else:
                    self.mutation_functions[name] = value
            elif "crossover" in name:
                if "global" in name:
                    self.global_crossover_kwargs = value
                else:
                    self.crossover_functions[name] = value
            elif "select" in name:
                if "global" in name:
                    self.global_select_kwargs = value
                else:
                    self.selection_functions[name] = value
            else:
                raise ValueError(
                    f"Each operator keyword must contain one of the following words: "
                    "['mutate', 'crossover', 'select']!"
                )
        else:
            raise ValueError(f"Section '{section}' is not a valid section for the config file!")
