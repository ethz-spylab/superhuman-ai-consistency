import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset

if "HF_HOME" in os.environ:
    CACHE_DIR = Path(os.environ["HF_HOME"]) / "cache"
else:
    CACHE_DIR = Path(__file__).absolute().parent.parent.parent / "cache"

if "DATASET_PATH" in os.environ:
    DATASET_PATH = Path(os.environ["DATASET_PATH"]) / "ECHR19"
else:
    DATASET_PATH = Path(__file__).absolute().parent.parent.parent.parent / "data/subsets"
CACHE_DIR = str(CACHE_DIR)
DATASET_PATH = str(DATASET_PATH)


ATTRIBUTES = {
    "ITEMID",
    "LANGUAGEISOCODE",
    "RESPONDENT",
    "BRANCH",
    "DATE",
    "DOCNAME",
    "IMPORTANCE",
    "CONCLUSION",
    "JUDGES",
    "TEXT",
    "VIOLATED_ARTICLES",
    "VIOLATED_PARAGRAPHS",
    "VIOLATED_BULLETPOINTS",
    "NON_VIOLATED_ARTICLES",
    "NON_VIOLATED_PARAGRAPHS",
    "NON_VIOLATED_BULLETPOINTS",
}


def load_echr19_sample(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract all attributes
    return {attr: data[attr] for attr in ATTRIBUTES}


def load_echr19(split: Optional[str] = None) -> Union[DatasetDict, Dataset, List]:
    if split is not None:
        assert split in [
            "train",
            "test",
            "validation",
            "toy",
        ], f"Invalid split. Must be one of 'train', 'test', 'validation'. Got {split}"

    if split == "toy":
        # We return a list of samples loaded from toydata/

        toy_data_dir = Path("toydata")
        all_files: List[str] = [str(f) for f in toy_data_dir.glob("**/*.json")]
        print("All files: ", all_files)
        assert len(all_files) > 0, "No files found in toydata/"
        samples = [load_echr19_sample(file_path) for file_path in all_files]
        return samples

    else:
        dataset = load_dataset(DATASET_PATH, cache_dir=CACHE_DIR)

        if split is None:
            return dataset

        return dataset[split]


def load_echr19_subset(
    subset_type: str,
    subset_size: int,
    attention_weights: bool = False,
    name_suffix: Optional[str] = None,
):
    accepted_subset_types = [
        "low_confidence",
        "high_confidence",
        "random",
        "logit_equal_distribution",
    ]
    assert (
        subset_type in accepted_subset_types
    ), f"Invalid subset type. Must be one of {accepted_subset_types}. Got {subset_type}"

    # Build the dataset path
    if attention_weights:
        attention_suffix = "_with_attention_weights"
    else:
        attention_suffix = ""
    if name_suffix is None:
        name_suffix = ""
    dataset_name = (
        f"ECHR19_subset_{subset_type}_size_{subset_size}{attention_suffix}{name_suffix}.json"
    )
    dataset_path = Path(DATASET_PATH) / dataset_name

    # Make sure the dataset exists
    assert dataset_path.exists(), f"Dataset {dataset_path} does not exist"

    # Load the dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Return the dataset
    return data


def store_echr19_subset(
    subset: List[Dict[str, Any]],
    subset_type: str,
    attention_weights: bool = False,
    name_suffix: Optional[str] = None,
):
    accepted_subset_types = [
        "low_confidence",
        "high_confidence",
        "random",
        "logit_equal_distribution",
    ]
    assert (
        subset_type in accepted_subset_types
    ), f"Invalid subset type. Must be one of {accepted_subset_types}. Got {subset_type}"

    # Build the dataset path
    subset_size = len(subset)
    if attention_weights:
        attention_suffix = "_with_attention_weights"
    else:
        attention_suffix = ""
    if name_suffix is None:
        name_suffix = ""
    dataset_name = (
        f"ECHR19_subset_{subset_type}_size_{subset_size}{attention_suffix}{name_suffix}.json"
    )
    dataset_path = Path(DATASET_PATH) / dataset_name

    # Store the dataset
    with open(dataset_path, "w") as f:
        json.dump(subset, f, indent=4)


def store_echr19(samples: List[Dict[str, Any]], split: str, path: str) -> None:
    split_map = {"train": "EN_train", "test": "EN_test", "dev": "EN_dev"}
    assert split in split_map, f"Invalid split. Must be one of 'train', 'test', 'dev'. Got {split}"

    # Create the result folder if it doesn't exist yet
    save_path = Path(path) / split_map[split]
    save_path.mkdir(parents=True, exist_ok=True)

    # Store each sample in an individual json file
    some_files_already_exist = False
    for sample in samples:
        file_name = sample["ITEMID"] + ".json"
        file_path = save_path / file_name

        # Check if the file already exists and prompt the user to overwrite it
        if file_path.exists():
            if not some_files_already_exist:
                print(
                    f"WARNING: Some files already exist in {save_path}. "
                    "Proceeding will overwrite the existing files. Continue? [y/n]"
                )
                answer = input()
                if answer.lower() != "y":
                    print("Aborting...")
                    return
                some_files_already_exist = True

        # Create the file and store the sample
        with open(file_path, "w") as f:
            json.dump(sample, f, indent=4)
