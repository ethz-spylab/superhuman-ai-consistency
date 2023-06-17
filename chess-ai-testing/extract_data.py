import pandas as pd
import argparse


def extract_data(input_path: str, output_path: str):
    fen_column_candidates = ["fen", "fen1"]
    df = pd.read_csv(input_path)

    # Find the column with the FENs
    fen_column = None
    for candidate in fen_column_candidates:
        if candidate in df.columns:
            fen_column = candidate
            break

    if fen_column is None:
        raise ValueError("Could not find a column with FENs")

    # Extract the FENs
    fens = df[fen_column]

    # Store the FENs in a file without the header and index
    fens.to_csv(output_path, header=False, index=False)


def check_equal(input_path1: str, input_path2: str):
    # Load the two dataframes
    # Both dataframes contain no index and no header
    df1 = pd.read_csv(input_path1, header=None)
    df2 = pd.read_csv(input_path2, header=None)

    # Check that both dataframes only have one column
    assert len(df1.columns) == 1
    assert len(df2.columns) == 1

    # Name the columns "fen"
    df1.columns = ["fen"]
    df2.columns = ["fen"]

    # Check that both dataframes have the same number of rows
    assert len(df1) == len(df2)

    # Order the rows of both dataframes
    df1 = df1.sort_values(by=["fen"])
    df2 = df2.sort_values(by=["fen"])

    # Reset the index of both dataframes
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # Check that both dataframes are equal
    assert df1.equals(df2), "The two files are not equal"

    print("The two files are equal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--input_file2", type=str, required=False)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--experiment", type=str, default="extract_data", choices=["extract_data", "check_equal"]
    )

    args = parser.parse_args()

    if args.experiment == "extract_data":
        extract_data(args.input_file, args.output_file)
    elif args.experiment == "check_equal":
        assert args.input_file2 is not None, "Need to provide a second input file"
        check_equal(args.input_file, args.input_file2)
