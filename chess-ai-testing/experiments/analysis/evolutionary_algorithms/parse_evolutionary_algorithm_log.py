from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

LOG_FOLDER = Path(__file__).parent.parent.parent.parent / "logs"

if __name__ == "__main__":
    # Interesting: best_fitness, run_time, num_generations, unique individual fraction
    #############################
    ##         CONFIG          ##
    #############################
    # log_file_name = Path("logs_evolutionary_algorithm_reweighted_probabilities_4.txt")
    # log_file_name = Path("logs_evolutionary_algorithm_cellular_1x720.txt")
    # log_file_name = Path("logs_evolutionary_algorithm_cellular_6x120.txt")
    # log_file_name = Path("logs_evolutionary_algorithm_cellular_12x60.txt")
    log_file_name = Path("logs_evolutionary_algorithm_cellular_24x30.txt")

    run_delimiter = "Starting run"

    # Mapping keywords appearing in the log file to better-readable names and the type of graph to plot
    metric_dict: Dict[str, Tuple[str, str]] = {
        "best_fitness = ": ("Best fitness", "line"),
        "Generation": ("Generation", "bar"),
        "Number of unique individuals:": ("Number of unique individuals", "line"),
        "Number of evaluations:": ("Number of evaluations", "bar"),
    }
    #############################
    ##      END OF CONFIG      ##
    #############################

    # Read the log file
    with open(LOG_FOLDER / log_file_name, "r") as f:
        logs = f.readlines()

    data: Dict[str, List[List[Any]]] = {metric: [] for metric in metric_dict}

    temp_data: Dict[str, List[Any]] = {}

    # Parse the log file
    for line in logs:
        if line.startswith(run_delimiter):
            # End of a run
            if temp_data:
                for metric in metric_dict:
                    if metric == "Generation":
                        data[metric].append(len(temp_data[metric]))
                    else:
                        data[metric].append(temp_data[metric])

            # Start a new run
            for metric in metric_dict:
                temp_data[metric] = []

        for name, (nice_name, plot_type) in metric_dict.items():
            if name in line:
                # Only keep the part of the line after the token
                line = line.split(name)[1].strip()

                # Extract the number and convert it to a float
                value = float(line.split(" ")[0])

                if plot_type == "bar" and name != "Generation":
                    temp_data[name] = value
                else:
                    temp_data[name].append(value)

    # Append the last run
    for metric in metric_dict:
        if metric == "Generation":
            data[metric].append(len(temp_data[metric]))
        else:
            data[metric].append(temp_data[metric])

    # Create a plot with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Increase the general font size
    plt.rcParams.update({"font.size": 18})

    # Plot the data
    for i, metric in enumerate(metric_dict):
        # Get the name and type of graph
        name, graph_type = metric_dict[metric]

        # Get the axis to plot on
        ax = axs[i // 2, i % 2]
        ax.grid(which="both", axis="both")

        # Plot the data
        if graph_type == "line":
            for index, run in enumerate(data[metric]):
                ax.plot(run, label=f"Run {index + 1}")
                ax.legend()
        elif graph_type == "bar":
            ax.bar(range(len(data[metric])), data[metric])

        # Set the title
        ax.set_title(name)

    fig.suptitle(log_file_name)

    # Show the plot
    plt.show()
