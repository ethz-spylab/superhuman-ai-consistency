# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""Template taken from: https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py"""


import csv
import json
import os
from pathlib import Path

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """\
Dataset for legal NLP tasks. Contains ~11,000 cases from the European Court of Human Rights.
The cases are split into a list of legal facts which are annotated with a binary label
indicating whether the case contains a violation of the European Convention on Human Rights, 
as well as multi-label annotations corresponding to the specific articles of the convention
that are violated.
"""

# Link to an official homepage for the dataset here
_HOMEPAGE = "https://archive.org/details/ECHR-ACL2019"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {}
_URLS["all_data"] = _URLS[
    "all_data_anonymized"
] = "https://archive.org/download/ECHR-ACL2019/ECHR_Dataset.zip"

_SPLIT_NAMES = {
    "all_data": {
        "train": "EN_train",
        "dev": "EN_dev",
        "test": "EN_test",
    },
    "all_data_anonymized": {
        "train": "EN_train_Anon",
        "dev": "EN_dev_Anon",
        "test": "EN_test_Anon",
    },
}


class ECHR19(datasets.GeneratorBasedBuilder):
    """Dataset for legal NLP tasks. Contains ~11,000 cases from the European Court of Human Rights.
    The cases are split into a list of legal facts which are annotated with a binary label
    indicating whether the case contains a violation of the European Convention on Human Rights,
    as well as multi-label annotations corresponding to the specific articles of the convention
    that are violated."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="all_data",
            version=VERSION,
            description="Contains all the case data from the ECHR dataset.",
        ),
        datasets.BuilderConfig(
            name="all_data_anonymized",
            version=VERSION,
            description=(
                "Contains all the case data from the ECHR dataset where"
                " all named entities have been replaced with anonymized tokens."
            ),
        ),
    ]

    DEFAULT_CONFIG_NAME = (  # It's not mandatory to have a default configuration. Just use one if it make sense.
        "all_data"
    )

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "ITEMID": datasets.Value("string"),
                "LANGUAGEISOCODE": datasets.Value("string"),
                "RESPONDENT": datasets.Value("string"),
                "BRANCH": datasets.Value("string"),
                "DATE": datasets.Value("int32"),
                "DOCNAME": datasets.Value("string"),
                "IMPORTANCE": datasets.Value("string"),
                "CONCLUSION": datasets.Value("string"),
                "JUDGES": datasets.Value("string"),
                "TEXT": datasets.Sequence(datasets.Value("string")),
                "VIOLATED_ARTICLES": datasets.Sequence(datasets.Value("string")),
                "VIOLATED_PARAGRAPHS": datasets.Sequence(datasets.Value("string")),
                "VIOLATED_BULLETPOINTS": datasets.Sequence(datasets.Value("string")),
                "NON_VIOLATED_ARTICLES": datasets.Sequence(datasets.Value("string")),
                "NON_VIOLATED_PARAGRAPHS": datasets.Sequence(datasets.Value("string")),
                "NON_VIOLATED_BULLETPOINTS": datasets.Sequence(datasets.Value("string")),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=split,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, _SPLIT_NAMES[self.config.name][split_str]),
                    "split": split_str,
                },
            )
            for split_str, split in zip(
                ["train", "dev", "test"],
                [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST],
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath: str, split: str):
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # The filepath is actually a directory containing the files for each split
        # Iterate over all the files in the directory
        for file in Path(filepath).iterdir():
            with open(file, encoding="utf-8") as f:
                data = json.load(f)

            yield data["ITEMID"], {
                "ITEMID": data["ITEMID"],
                "LANGUAGEISOCODE": data["LANGUAGEISOCODE"],
                "RESPONDENT": data["RESPONDENT"],
                "BRANCH": data["BRANCH"],
                "DATE": data["DATE"],
                "DOCNAME": data["DOCNAME"],
                "IMPORTANCE": data["IMPORTANCE"],
                "CONCLUSION": data["CONCLUSION"],
                "JUDGES": data["JUDGES"],
                "TEXT": data["TEXT"],
                "VIOLATED_ARTICLES": data["VIOLATED_ARTICLES"],
                "VIOLATED_PARAGRAPHS": data["VIOLATED_PARAGRAPHS"],
                "VIOLATED_BULLETPOINTS": data["VIOLATED_BULLETPOINTS"],
                "NON_VIOLATED_ARTICLES": data["NON_VIOLATED_ARTICLES"],
                "NON_VIOLATED_PARAGRAPHS": data["NON_VIOLATED_PARAGRAPHS"],
                "NON_VIOLATED_BULLETPOINTS": data["NON_VIOLATED_BULLETPOINTS"],
            }
