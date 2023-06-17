import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, Union


def get_git_infos() -> Tuple[str, str, bool]:
    """Get the current branch, commit hash and uncommitted changes. If any of the
    commands fail, the corresponding value is set to "unknown".

    Returns:
        Tuple[str, str, bool]: The current branch, commit hash and a boolean value indicating
            whether there are uncommitted changes.
    """
    try:
        # Get the current git branch
        branch_bytes = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        branch = branch_bytes.decode("utf-8").strip()

        # Get the current git commit hash
        commit_hash_bytes = subprocess.check_output(["git", "rev-parse", "HEAD"])
        commit_hash = commit_hash_bytes.decode("utf-8").strip()

        # Check if there are uncommitted changes
        uncommitted_changes_bytes = subprocess.check_output(["git", "status", "--porcelain"])
        uncommitted_changes = uncommitted_changes_bytes.decode("utf-8").strip()
        uncommitted_changes = bool(uncommitted_changes)
    except subprocess.CalledProcessError:
        branch = "unknown"
        commit_hash = "unknown"
        uncommitted_changes = False

    return branch, commit_hash, uncommitted_changes


def store_experiment_params(
    namespace: argparse.Namespace,
    result_file_path: Union[str, Path],
    source_file_path: Union[str, Path],
):
    # Get the git infos
    branch, commit_hash, uncommitted_changes = get_git_infos()

    # Convert the namespace to a dictionary
    argument_dict = vars(namespace)

    # Write the configuration to a file
    with open(result_file_path, "w") as result_file:
        result_file.write("CONFIGURATION:\n")
        result_file.write(f"experiment file = {source_file_path}\n")
        result_file.write(f"git branch = {branch}\n")
        result_file.write(f"git commit hash = {commit_hash}\n")
        result_file.write(f"uncommitted changes = {uncommitted_changes}\n")
        for key, value in argument_dict.items():
            result_file.write(f"{key} = {value}\n")
        result_file.write("\n")


def get_experiment_params_dict(
    namespace: argparse.Namespace,
    source_file_path: Union[str, Path],
) -> Dict[str, Any]:
    # Get the git infos
    branch, commit_hash, uncommitted_changes = get_git_infos()

    # Convert the namespace to a dictionary
    argument_dict = vars(namespace)

    # Add the git infos to the argument dictionary
    argument_dict["experiment file"] = source_file_path
    argument_dict["git branch"] = branch
    argument_dict["git commit hash"] = commit_hash
    argument_dict["uncommitted changes"] = uncommitted_changes

    return argument_dict
