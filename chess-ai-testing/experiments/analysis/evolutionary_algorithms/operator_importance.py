from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rl_testing.evolutionary_algorithms.crossovers import CrossoverName
from rl_testing.evolutionary_algorithms.mutations import MutationName

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"


if __name__ == "__main__":
    wandb_file_name = "wandb_sweep_result_2023-02-15.csv"
    importance_metric_name = "best_fitness_value_avg"
    minimum_importance_constant = 0.05

    # Must be a subset of ["mutation", "crossover"]
    operator_types = ["crossover"]

    # Compute the mutation function names
    mutation_names = [mutation_name.name.lower() for mutation_name in MutationName]

    # Compute the crossover function names
    crossover_names = [crossover_name.name.lower() for crossover_name in CrossoverName]

    # Load the data
    data = pd.read_csv(DATA_PATH / wandb_file_name)

    name_list, importance_list = [], []

    # Create a list of operator names
    operator_names = []
    if "mutation" in operator_types:
        operator_names += mutation_names
    if "crossover" in operator_types:
        operator_names += crossover_names

    # For each operator compute the average importance when it is used and when it is not used
    for operator_name in operator_names:
        grouped_data = data.groupby(operator_name)[importance_metric_name].mean()
        name_list.append(operator_name)
        importance_list.append(grouped_data[True] - grouped_data[False])

    # Plot the results as a bar plot. If the bar is negative, it should be red, otherwise it should be green.
    plt.bar(
        name_list,
        importance_list,
        color=["red" if importance < 0 else "green" for importance in importance_list],
    )

    plt.grid(
        which="both",
    )

    # Increase general matplotlib font size
    plt.rcParams.update({"font.size": 20})

    plt.xticks(rotation=90, fontsize=14)
    plt.ylabel("Importance", fontsize=18)
    plt.title("Operator Importance")
    plt.subplots_adjust(bottom=0.3)

    # plt.show()
    # plt.savefig("operator_importance.png", dpi=300, bbox_inches="tight")

    # Rescale the importance values to be greater or equal to 0
    # Also add a constant to avoid having importance values equal to 0
    min_importance = min(importance_list)
    importance_list = [
        importance - min_importance + minimum_importance_constant for importance in importance_list
    ]

    # Normalize the importance values
    importance_list = [importance / sum(importance_list) for importance in importance_list]

    # Print the importance values and the corresponding operator names
    for name, importance in zip(name_list, importance_list):
        print(f"{name:30}: {importance}")
