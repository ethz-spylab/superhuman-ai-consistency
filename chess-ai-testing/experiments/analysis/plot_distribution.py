import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from load_results import (
    compare_columns_and_filter,
    compute_differences,
    flip_q_values,
    load_data,
)

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, help="Path to result file", required=True)  # noqa
    parser.add_argument("--column_name1", type=str, help="Name of first column to use for comparison", required=False, default="parent_score")  # noqa
    parser.add_argument("--column_name2", type=str, help="Name of second column to use for comparison", required=False, default="child_score")  # noqa
    parser.add_argument("--column_names", nargs="*", help="List of columns to use for comparison if there are more than 2. This parameter has priority over 'column_name1' and 'column_name2'", required=False, default=None)  # noqa
    parser.add_argument("--column_difference_name", type=str, help="If the difference between two columns has already been computed, use this argument to specify the name of the column storing the difference", required=False, default=None)  # noqa
    parser.add_argument("--q_vals_to_flip", nargs="*", help="List of columns where the stored q-values should be flipped (multiplied by -1)", required=False, default=[])  # noqa
    parser.add_argument("--column1_for_compare_and_prefilter", type=str, help="Name of the first column to use for comparison and prefiltering (Done before computing difference)", required=False, default=None)  # noqa
    parser.add_argument("--column2_for_compare_and_prefilter", type=str, help="Name of the second column to use for comparison and prefiltering (Done before computing difference)", required=False, default=None)  # noqa
    parser.add_argument("--compare_string", type=str, help="String to use for comparison", required=False, choices=["==", "!=", "<", "<=", ">", ">="], default=None)  # noqa
    parser.add_argument("--x_limit_min", type=float, help="Minimum value for x-axis limit", required=False, default=0)  # noqa
    parser.add_argument("--x_limit_max", type=float, help="Maximum value for x-axis limit", required=False, default=2)  # noqa
    parser.add_argument("--y_limit_min", type=float, help="Minimum value for y-axis limit", required=False, default=None)  # noqa
    parser.add_argument("--y_limit_max", type=float, help="Maximum value for y-axis limit", required=False, default=None)  # noqa
    parser.add_argument("--title", type=str, nargs="+", help="Title for plot", required=False, default="")  # noqa
    parser.add_argument("--save_csv", action="store_true", help="Save the resulting dataframe to a file", required=False, default=False)  # noqa
    parser.add_argument("--save_plot", action="store_true", help="Save the resulting plots to a file", required=False, default=False)  # noqa
    parser.add_argument("--show_plot", action="store_true", help="Show the resulting plots", required=False, default=False)  # noqa
    # fmt: on
    return parser.parse_args()


def differences_density_plot(
    dataframe: pd.DataFrame,
    column_name1: Optional[str] = None,
    column_name2: Optional[str] = None,
    column_names: Optional[List[str]] = None,
    x_limits: Tuple[float, float] = (0, 2),
    y_limits: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_plot: bool = True,
    save_plot: bool = False,
    save_plot_path: Optional[Path] = None,
) -> pd.DataFrame:
    if save_plot:
        assert save_plot_path is not None, "If save_plot is True, save_plot_path must be specified!"

    if column_names is not None:
        dataframe = compute_differences(dataframe, *column_names)
    elif column_name1 is not None and column_name2 is not None:
        dataframe = compute_differences(dataframe, column_name1, column_name2)
    else:
        dataframe = dataframe.sort_values(by="difference", ascending=False)

    print(dataframe[:100].to_string())
    plt.hist(dataframe["difference"], bins=1000, range=x_limits)

    if y_limits is not None:
        plt.ylim(bottom=y_limits[0], top=y_limits[1])

    max_difference = dataframe["difference"].max()
    font = {"size": 16}
    matplotlib.rc("font", **font)
    plt.xlabel("Value difference", fontdict=font)
    plt.ylabel("Amount", fontdict=font)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.yscale("log")
    print(max_difference)
    _, y_max = plt.gca().get_ylim()
    plt.vlines([max_difference], ymin=[0], ymax=y_max, colors=["red"])
    if title is not None:
        plt.title(title)

    if show_plot:
        plt.show()

    if save_plot:
        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")

    return dataframe


if __name__ == "__main__":

    # Parse the arguments
    args = parse_args()

    # Extract the arguments
    title = " ".join(args.title)
    result_path = Path(args.result_path)
    x_limits = (args.x_limit_min, args.x_limit_max)
    y_limits = (args.y_limit_min, args.y_limit_max)
    if x_limits == (None, None):
        x_limits = None
    if y_limits == (None, None):
        y_limits = None
    columns_to_compare = (
        args.column1_for_compare_and_prefilter,
        args.column2_for_compare_and_prefilter,
    )
    if columns_to_compare == (None, None):
        columns_to_compare = None
    compare_string = args.compare_string

    # Check if the result_path is absolute
    if not result_path.is_absolute():
        result_path = PROJECT_ROOT / result_path

    # Load the stored experiment data
    dataframe, _ = load_data(result_path=result_path)

    # Check if the difference column is already present
    if args.column_name1 is None or args.column_name2 is None:
        assert (
            args.column_difference_name is not None
        ), "If 'column_name1' or 'column_name2' is None, 'column_difference_name' must be set."

        dataframe["difference"] = dataframe[args.column_difference_name]

    # Check if there are any result columns whose q-values should be flipped
    for column_name in args.q_vals_to_flip:
        dataframe = flip_q_values(dataframe, column_name=column_name)

    # Check if there are any columns that should be used for prefiltering the data
    if columns_to_compare:
        dataframe = compare_columns_and_filter(
            dataframe, *columns_to_compare, compare_string=compare_string
        )

    if args.save_plot:
        save_plot_path = result_path.with_suffix("").with_suffix(".png")
    else:
        save_plot_path = None

    # Plot a histogram of the differences
    dataframe = differences_density_plot(
        dataframe=dataframe,
        column_name1=args.column_name1,
        column_name2=args.column_name2,
        column_names=args.column_names,
        x_limits=x_limits,
        y_limits=y_limits,
        title=title,
        show_plot=args.show_plot,
        save_plot=args.save_plot,
        save_plot_path=save_plot_path,
    )

    # Store the dataframe
    if args.save_csv:
        store_path = str(result_path.with_suffix("")) + "_differences_sorted.csv"
        dataframe.to_csv(store_path)
