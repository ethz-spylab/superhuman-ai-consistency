import argparse
import json
from datetime import datetime
import copy
from pathlib import Path
import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt

import numpy as np
from legal_testing.util.plot_paraphrases import plot_paraphrases

import matplotlib.ticker as ticker
import matplotlib.scale as mscale
import pprint

RESULT_DIR = Path(__file__).absolute().parent / "results/plots"

ACCEPTED_PLOT_TYPES = ["summary_plot_success", "summary_plot_sensitivity"]


class LogitScale(mscale.ScaleBase):
    """Custom logit scale for matplotlib."""

    name = "my_logit"

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis)
        self.nonsingular = kwargs.get("nonsingular", (-0.25, 0.25))

    def get_transform(self):
        return self.LogitTransform(self.nonsingular)

    def set_default_locators_and_formatters(self, axis):
        vals = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        formats = [
            "$10^{-2}$",
            "$10^{-1}$",
            "1/4",
            "1/2",
            "3/4",
            "$1-10^{-1}$",
            "$1-10^{-2}$",
        ]

        axis.set_major_locator(ticker.LogLocator(subs=vals))
        axis.set_minor_locator(ticker.LogLocator(subs=[]))
        # axis.set_major_formatter(ticker.ScalarFormatter())

        axis.set_major_formatter(lambda x, pos: formats[vals.index(x)])

    class LogitTransform(mscale.Transform):
        input_dims = output_dims = 1

        def __init__(self, nonsingular):
            mscale.Transform.__init__(self)
            self.nonsingular = nonsingular

        def transform_non_affine(self, a):
            a = np.array(a)
            return np.log(a / (1.0 - a))

        def inverted(self):
            return LogitScale.InvertedLogitTransform(self.nonsingular)

    class InvertedLogitTransform(mscale.Transform):
        input_dims = output_dims = 1

        def __init__(self, nonsingular):
            mscale.Transform.__init__(self)
            self.nonsingular = nonsingular

        def transform_non_affine(self, a):
            a = np.array(a)
            return 1.0 / (1.0 + np.exp(-a))

        def inverted(self):
            return LogitScale.LogitTransform(self.nonsingular)


def compute_violation_statistics(
    cases: Dict[str, Dict[str, Any]], bins: List[float]
) -> Dict[str, Dict[str, Any]]:
    """Given a dictionary of results of paraphrasing attacks, compute some statistics about the
    results. (e.g. number of successful attacks, average probability difference, etc.)

    Args:
        cases (Dict[str, Dict[str, Any]]): A dictionary of results of paraphrasing attacks
        bins (List[float]): The bins to be used for computing the probability difference bins

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing the computed statistics
    """
    attack_types = ["PARAPHRASES_RANDOM", "PARAPHRASES_MOST_IMPORTANT_FACT"]

    successful_attacks = {attack_type: 0 for attack_type in attack_types}
    successful_from_negative_to_positive = {attack_type: 0 for attack_type in attack_types}
    successful_from_positive_to_negative = {attack_type: 0 for attack_type in attack_types}

    probability_differences = {attack_type: [] for attack_type in attack_types}
    log_odds_differences = {attack_type: [] for attack_type in attack_types}

    def log_odds(probability):
        return np.log(probability / (1 - probability)) if probability != 0 else -np.inf

    max_difference = {attack_type: 0 for attack_type in attack_types}
    max_difference_info = {attack_type: None for attack_type in attack_types}
    positive_original_predictions = {attack_type: 0 for attack_type in attack_types}
    negative_original_predictions = {attack_type: 0 for attack_type in attack_types}

    # Iterate through all cases
    for sample_id in cases:
        sample = cases[sample_id]

        # Extract the original logit and prediction
        original_logit = sample["original_logit"]
        original_prediction = sample["original_prediction"]

        # Iterate through all results
        for result in sample["results"]:
            attack_type = result["attack_type"]

            # Extract the modified logit and prediction
            modified_logit = result["logit"]
            modified_prediction = result["prediction"]

            # Check if the prediction changed
            if original_prediction != modified_prediction:
                successful_attacks[attack_type] += 1

                if original_prediction == 0:
                    successful_from_negative_to_positive[attack_type] += 1
                else:
                    successful_from_positive_to_negative[attack_type] += 1

            # Compute the probability difference
            probability_differences[attack_type].append(abs(original_logit - modified_logit))

            # Check if this is the maximum difference
            if abs(original_logit - modified_logit) > max_difference[attack_type]:
                max_difference[attack_type] = abs(original_logit - modified_logit)
                max_difference_info[attack_type] = {
                    "original_probability": original_logit,
                    "modified_probability": modified_logit,
                }

            # Compute the log odds difference
            log_odds_differences[attack_type].append(
                abs(log_odds(original_logit) - log_odds(modified_logit))
            )

            # Check if the original prediction was positive or negative
            if original_prediction == 0:
                negative_original_predictions[attack_type] += 1
            else:
                positive_original_predictions[attack_type] += 1

    # Store some basic statistics
    statistics = {
        "max_difference": max_difference,
        "max_difference_info": max_difference_info,
        "successful_attacks": successful_attacks,
        "positive_original_predictions": positive_original_predictions,
        "negative_original_predictions": negative_original_predictions,
        "successful_from_negative_to_positive": successful_from_negative_to_positive,
        "successful_from_positive_to_negative": successful_from_positive_to_negative,
    }

    statistics["average_probability_differences"] = {}
    statistics["average_log_odds_differences"] = {}
    statistics["probability_differences_bins"] = {}

    # Compute some basic statistics
    for attack_type in attack_types:
        # Compute the average probability difference
        statistics["average_probability_differences"][attack_type] = np.mean(
            probability_differences[attack_type]
        )
        statistics["average_log_odds_differences"][attack_type] = np.mean(
            log_odds_differences[attack_type]
        )

        # Compute the probability difference bins
        statistics["probability_differences_bins"][attack_type] = {
            f"[{bins[i]}, {bins[i+1]})": sum(
                [
                    1
                    for probability_difference in probability_differences[attack_type]
                    if probability_difference >= bins[i] and probability_difference < bins[i + 1]
                ]
            )
            for i in range(len(bins) - 1)
        }

    return statistics


def filter_results(
    result_data: Dict[str, Any],
    attack_type: Literal["PARAPHRASES_RANDOM", "PARAPHRASES_MOST_IMPORTANT_FACT", "all"] = "all",
    num_facts_per_case: Union[int, Literal["all"]] = "all",
    num_paraphrases_per_fact: Union[int, Literal["all"]] = "all",
) -> Dict[str, Any]:
    """Given a dictionary of results of paraphrasing attacks, filter the results according to the
    given parameters. This allows to examine specific subsets of the data.

    Args:
        result_data (Dict[str, Any]): A dictionary of results of paraphrasing attacks
        attack_type (Literal["PARAPHRASES_RANDOM", "PARAPHRASES_MOST_IMPORTANT_FACT", "all"],
            optional): Which attack type to include in the results. Defaults to "all".
        num_facts_per_case (Union[int, Literal["all"]], optional): The number of facts per case
            to be included in the filtered subset. Defaults to "all".
        num_paraphrases_per_fact (Union[int, Literal["all"]], optional): The number of paraphrases
            per fact to be included in the filtered subset. Defaults to "all".

    Returns:
        Dict[str, Any]: A dictionary containing the filtered results
    """
    # Create a copy of the result data
    result_data_copy = copy.deepcopy(result_data)

    # Iterate over all samples
    for item_id in result_data_copy:
        sample = result_data_copy[item_id]
        results = sample["results"]

        # Shuffle the paraphrases
        random.shuffle(results)

        # Filter out all results which do not have the correct attack type
        if attack_type != "all":
            results = [result for result in results if result["attack_type"] == attack_type]

        # Make sure that at most 'num_facts_per_case' facts are included
        if num_facts_per_case != "all":
            # Get the unique "attack_index" values
            attack_indices_unique = list(set([result["attack_index"] for result in results]))

            if len(attack_indices_unique) > num_facts_per_case:
                # Filter out all results which do not have one of the 'num_facts_per_case' largest
                # attack indices
                results = [
                    result
                    for result in results
                    if result["attack_index"] in attack_indices_unique[-num_facts_per_case:]
                ]

        # Make sure that at most 'num_paraphrases_per_fact' paraphrases are included per fact
        if num_paraphrases_per_fact != "all":
            # Get the unique "attack_index" values
            attack_indices_unique = list(set([result["attack_index"] for result in results]))
            paraphrase_counters = {attack_index: 0 for attack_index in attack_indices_unique}

            new_results = []
            for result in results:
                if paraphrase_counters[result["attack_index"]] < num_paraphrases_per_fact:
                    new_results.append(result)
                    paraphrase_counters[result["attack_index"]] += 1

            results = new_results

        # Store the filtered results
        result_data_copy[item_id]["results"] = results

    return result_data_copy


def plot_sensitivity_summary(
    results: Dict[str, Any],
    title: Optional[str] = None,
    logit_axis: bool = False,
    save_plot: bool = False,
    file_name: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """Create a scatter plot of the sensitivity of a model to different attacks.

    Args:
        results (Dict[str, Any]): A dictionary of results of paraphrasing attacks
        title (Optional[str], optional): The title of the plot. Defaults to None.
        logit_axis (bool, optional): Whether to use logits for the axis or not. Defaults to False.
        save_plot (bool, optional): Whether to save the plot or not. Defaults to False.
        file_name (Optional[str], optional): The name of the file to save the plot to.
            Defaults to None.
        show_plot (bool, optional): Whether to show the plot or not. Defaults to True.
    """
    # Extract original logit values and success rates
    values: List[Tuple[float, float]] = []

    for item_id in results:
        sample = results[item_id]
        original_logit = sample["original_logit"]
        new_logits = [sample["results"][i]["logit"] for i in range(len(sample["results"]))]
        # new_logits = [sample["results"][i]["logit"] for i in range(1)]
        values.extend([(original_logit, new_logit) for new_logit in new_logits])

    # Get all values where the classification changed
    classification_changed_values = [
        (original_logit, new_logit)
        for original_logit, new_logit in values
        if original_logit > 0.5 and new_logit < 0.5 or original_logit < 0.5 and new_logit > 0.5
    ]

    x_values = np.array([x[0] for x in values])
    y_values = np.array([x[1] for x in values])

    x_values_changed = np.array([x[0] for x in classification_changed_values])
    y_values_changed = np.array([x[1] for x in classification_changed_values])

    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)

    # Register the custom scale
    mscale.register_scale(LogitScale)

    if logit_axis:
        plt.yscale("my_logit")
        plt.xscale("my_logit")

        # Create the scatterplot with logit scale
        plt.xticks([0.01, 0.1, 0.5, 0.9, 0.99])
        plt.yticks([0.01, 0.1, 0.5, 0.9, 0.99])

    else:
        plt.xticks([0, 0.25, 0.5, 0.75, 1])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])

    # Plot the results
    # Draw a black, dashed line from (0,0) to (1,1)

    # plt.rcParams["figure.figsize"] = [4, 3]
    plt.grid(which="both", axis="both")

    plt.axline((0.01, 0.01), (0.99, 0.99), linestyle="--", color="black", zorder=5)
    plt.scatter(x_values, y_values, s=2, zorder=10)

    # Plot circles around the values where the classification changed

    for index in range(len(x_values_changed)):
        plt.scatter(
            x_values_changed[index],
            y_values_changed[index],
            s=25,
            facecolors="none",
            edgecolors="red",
            zorder=11,
        )

    plt.xlabel("P(violation in original case)", fontsize=18)
    plt.ylabel("P(violation in paraphrased case)", fontsize=18)

    if logit_axis:
        plt.ylim(0.01, 0.99)
        plt.xlim(0.01, 0.99)
    else:
        plt.ylim(0, 1)
        plt.xlim(0, 1)

    if title is not None:
        plt.title(title, fontsize=20)
    # plt.hlines(0.5, 0, 1, colors="red")
    if show_plot:
        plt.show()

    if save_plot:
        plt.tight_layout()
        current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if file_name is None:
            title = title + "_" if title is not None else ""
            save_path = RESULT_DIR / f"Sensitivity_plot_{title}{current_date_time}.png"
        else:
            save_path = RESULT_DIR / f"{file_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_success_summary(
    results: Dict[str, Any],
    title: Optional[str] = None,
    save_plot: bool = False,
    show_plot: bool = False,
) -> None:
    """Create a bar plot of the success rate of a model to different attacks.

    Args:
        results (Dict[str, Any]): A dictionary of results of paraphrasing attacks
        title (Optional[str], optional): The title of the plot. Defaults to None.
        save_plot (bool, optional): Whether to save the plot or not. Defaults to False.
        show_plot (bool, optional): Whether to show the plot or not. Defaults to True.
    """
    # Extract original logit values and success rates
    values: Tuple[float, float] = []

    for item_id in results:
        sample = results[item_id]
        original_logit = sample["original_logit"]
        original_prediction = sample["original_prediction"]
        num_successful_attacks = sum(
            sample["results"][i]["prediction"] != original_prediction
            for i in range(len(sample["results"]))
        )
        total_attacks = len(sample["results"])
        success_rate = 0 if total_attacks == 0 else num_successful_attacks / total_attacks
        values.append((original_logit, success_rate))

    # Order the values by the original logit in ascending order
    values = sorted(values, key=lambda x: x[0])

    # Plot the results
    plt.rcParams["figure.figsize"] = [15, 10]
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)
    plt.bar([x[0] for x in values], [x[1] for x in values], width=0.01)
    plt.xlabel("Original logit", fontsize=24)
    plt.ylabel("Attack success rate (%)", fontsize=24)
    plt.ylim(0, 1)
    if title is not None:
        plt.title(title, fontsize=24)
    plt.grid(which="both", axis="both")

    if show_plot:
        plt.show()

    if save_plot:
        plt.tight_layout()
        current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        title = title + "_" if title is not None else ""
        save_path = RESULT_DIR / f"Success_plot_{title}{current_date_time}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.close()


def plot_paraphrasing_facts_summary(
    result_file: str,
    plot_type: str,
    num_facts_per_case: Union[int, Literal["all"]] = "all",
    num_paraphrases_per_fact: Union[int, Literal["all"]] = "all",
    logit_axis: bool = False,
    save_plot: bool = False,
    show_plot: bool = True,
) -> None:
    """Plot the results of paraphrasing attacks on the ECHR dataset.

    Args:
        result_file (str): The result file to be used
        plot_type (str): The type of plot to be created. Must be one of ["summary_plot_success",
            "summary_plot_sensitivity"]
        num_facts_per_case (Union[int, Literal["all"]], optional): The number of facts
            per case to be included in the filtered subset. Defaults to "all".
        num_paraphrases_per_fact (Union[int, Literal["all"]], optional): The number of
            paraphrases per fact to be included in the filtered subset. Defaults to "all".
        logit_axis (bool, optional): Whether to use logits for the axis or not. Defaults to False.
        save_plot (bool, optional): Whether to save the plot or not. Defaults to False.
        show_plot (bool, optional): Whether to show the plot or not. Defaults to True.
    """
    assert (
        plot_type in ACCEPTED_PLOT_TYPES
    ), f"Invalid plot type. Must be one of {ACCEPTED_PLOT_TYPES}"

    # Open result file
    with open(result_file, "r") as file:
        results_complete: Dict[str, Any] = json.load(file)

    pp = pprint.PrettyPrinter(indent=4)

    # Iterate the different attack types
    for attack_type in ["PARAPHRASES_RANDOM", "PARAPHRASES_MOST_IMPORTANT_FACT"]:
        # Filter the data
        results = filter_results(
            results_complete,
            attack_type=attack_type,
            num_facts_per_case=num_facts_per_case,
            num_paraphrases_per_fact=num_paraphrases_per_fact,
        )

        # Compute some statistics about the results
        statistics = compute_violation_statistics(
            results, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        print(f"Statistics for attack type {attack_type}:")
        pp.pprint(statistics)
        print()

        # Create the summary plot
        title = None
        if plot_type == "summary_plot_success":
            plot_success_summary(
                results,
                title=title,
                save_plot=save_plot,
                show_plot=show_plot,
            )
        elif plot_type == "summary_plot_sensitivity":
            file_name = (
                f"sensitivity_plot_{attack_type[12:].lower()}_{num_facts_per_case}"
                f"_facts_{num_paraphrases_per_fact}_paraphrases"
            )
            plot_sensitivity_summary(
                results,
                title=title,
                logit_axis=logit_axis,
                save_plot=save_plot,
                file_name=file_name,
                show_plot=show_plot,
            )


def plot_successful_paraphrases(
    original_data_path: str,
    result_path: str,
    save_plot: bool = False,
    show_plot: bool = True,
) -> None:
    """Create plots of all paraphrases that successfully changed the prediction.

    Args:
        original_data_path (str): The original data file used for extracting the original cases
        result_path (str): The result file storing the results of the paraphrasing attacks
        plot_type (str): The type of plot to be created. Must be one of ["summary_plot_success", "summary_plot_sensitivity"] # noqa
        save_plot (bool, optional): Whether to save the plot or not. Defaults to False.
        show_plot (bool, optional): Whether to show the plot or not. Defaults to True.
    """
    # Load original data
    with open(original_data_path, "r") as file:
        original_data_list = json.load(file)
        original_data = {
            sample["ITEMID"]: sample for sample in original_data_list if "ITEMID" in sample
        }

    # Open result file
    with open(result_path, "r") as file:
        results_complete: Dict[str, Any] = json.load(file)

    # Iterate the different attack types
    for attack_type in ["PARAPHRASES_RANDOM", "PARAPHRASES_MOST_IMPORTANT_FACT"]:
        # Create plots of all successful attacks of this type
        for item_index, item_id in enumerate(results_complete):
            print(f"Processing sample {item_index}/{len(results_complete)}")

            sample = results_complete[item_id]
            successful_attacks = [
                result
                for result in sample["results"]
                if result["attack_type"] == attack_type
                and result["prediction"] != sample["original_prediction"]
            ]

            for item_index, successful_attack in enumerate(successful_attacks):
                # Get the original sentence of the fact
                original_fact = original_data[item_id]["TEXT"][successful_attack["attack_index"]]

                # Plot the paraphrases
                if save_plot:
                    save_path = (
                        RESULT_DIR
                        / f"paraphrases_{item_id}_{attack_type[12:].lower()}_{item_index}.png"
                    )
                else:
                    save_path = None
                try:
                    plot_paraphrases(
                        sentence1=original_fact,
                        sentence2=successful_attack["paraphrase"],
                        save_plot=save_plot,
                        save_path=save_path,
                        show_plot=show_plot,
                    )
                except IndexError:
                    print(f"Failed to plot paraphrases for sample {item_id}")


def list_successful_attacks(
    result_file: str,
    attacking_fact_key: str,
    additional_result_fields: List[str] = [],
    order_by: str = "length",
) -> None:
    """List all successful attacks.

    Args:
        result_file (str): The result file to be used
        attacking_fact_key (str): The key of the attacking fact to be used
        additional_result_fields (List[str], optional): Additional fields to be printed.
            Defaults to [].
        order_by (str, optional): The field to order the successful attacks by.
            Defaults to "length".
    """
    assert order_by in ["length", "logit_difference"], "Invalid order_by value."

    # Open result file
    with open(result_file, "r") as file:
        results = json.load(file)

    successful_attacks: List[Dict[str, Any]] = []

    for index, item_id in enumerate(results):
        # Log progress
        print(f"Processing sample {index}/{len(results)}: {item_id}")

        sample = results[item_id]

        # Check if the attack was successful
        original_prediction = sample["original_prediction"]
        for result in sample["results"]:
            if result["prediction"] != original_prediction:
                # Copy the info above into this dictionary
                successful_attack = {
                    "ITEMID": item_id,
                    "original_prediction": original_prediction,
                    "new_prediction": result["prediction"],
                    "original_logit": sample["original_logit"],
                    "new_logit": result["logit"],
                    "attacking_fact": result[attacking_fact_key],
                }

                for additional_result_field in additional_result_fields:
                    successful_attack[additional_result_field] = result[additional_result_field]

                successful_attacks.append(successful_attack)

    # Sort the successful attacks by length
    if order_by == "length":
        successful_attacks = sorted(
            successful_attacks,
            key=lambda x: len(x["attacking_fact"]),
            reverse=False,
        )
    elif order_by == "logit_difference":
        successful_attacks = sorted(
            successful_attacks,
            key=lambda x: abs(x["original_logit"] - x["new_logit"]),
            reverse=True,
        )

    # Print the successful attacks
    for successful_attack in successful_attacks:
        print("=" * 30)
        for key in successful_attack:
            print(f"{key}: {successful_attack[key]}")
        print()


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, help="From which experiment type the data stems", default="paraphrasing_facts", choices=["paraphrasing_facts"]) # noqa
    parser.add_argument("--experiment_type", type=str, help="The type of analysis that shall be done with the experiment data.", default="summary_plot_success", choices=["summary_plot_success", "summary_plot_sensitivity","list_successful_attacks", "plot_successful_paraphrases"]) # noqa
    parser.add_argument("--num_facts_per_case", type=str, help="The number of facts per case to be included in the plot", default="all") # noqa
    parser.add_argument("--num_paraphrases_per_fact", type=str, help="The number of paraphrases per fact to be included in the plot", default="all") # noqa
    parser.add_argument("--result_path", type=str, help="The result file to be used", default=None) # noqa
    parser.add_argument("--original_data_path", type=str, help="The original data file to be used", default=None) # noqa
    parser.add_argument("--logit_axis", action="store_true", help="Whether to use logits for the axis or not", default=False) # noqa
    parser.add_argument("--order_by", type=str, help="The field to order the successful attacks by", default="length", choices=["length", "logit_difference"]) # noqa
    parser.add_argument("--save_plot", action="store_true", help="Whether to save the plot or not", default=False) # noqa
    parser.add_argument("--show_plot", action="store_true", help="Whether to show the plot or not", default=False) # noqa
    args = parser.parse_args()
    # fmt: on

    num_facts_per_case = args.num_facts_per_case
    if num_facts_per_case != "all":
        num_facts_per_case = int(args.num_facts_per_case)
    num_paraphrases_per_fact = args.num_paraphrases_per_fact
    if num_paraphrases_per_fact != "all":
        num_paraphrases_per_fact = int(args.num_paraphrases_per_fact)
    if "summary_plot" in args.experiment_type:
        plot_paraphrasing_facts_summary(
            result_file=args.result_path,
            plot_type=args.experiment_type,
            num_facts_per_case=num_facts_per_case,
            num_paraphrases_per_fact=num_paraphrases_per_fact,
            logit_axis=args.logit_axis,
            save_plot=args.save_plot,
            show_plot=args.show_plot,
        )

    elif args.experiment_type == "plot_successful_paraphrases":
        assert args.original_data_path is not None, "Original data path must be specified."
        plot_successful_paraphrases(
            original_data_path=args.original_data_path,
            result_path=args.result_path,
            save_plot=args.save_plot,
            show_plot=args.show_plot,
        )

    elif args.experiment_type == "list_successful_attacks":
        list_successful_attacks(
            result_file=args.result_path,
            attacking_fact_key="paraphrase",
            additional_result_fields=["attack_type", "attack_index"],
            order_by=args.order_by,
        )
