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
"""CameraTrap Crop dataset."""

import pandas as pd
import csv
import json
import os

import datasets
from datasets.tasks import ImageClassification

from PIL import Image

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""


# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed as an image classification problem. For classifying animal crops from Telangana AI Forest Challenge.
There are 4k+ image crops primarily over the following 4 categories: cows, goats, sambhar & chithal 
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "tai4species": "https://storage.googleapis.com/telangana-cameratraps/cameratraps4.zip",
    "tai20species": "https://storage.googleapis.com/telangana-cameratraps/cameratraps20.zip",
    "taiALLspecies": "https://storage.googleapis.com/telangana-cameratraps/cameratrapsALL.zip",
}

_NAMES20 = [
    "birds",
    "black naped hare",
    "chinkara",
    "chithal",
    "cows",
    "dhole",
    "domestic dogs",
    "gaur",
    "goat",
    "honey badger",
    "langur",
    "leopard",
    "macquae",
    "nilgai",
    "porcupine",
    "ring civet",
    "sambhar",
    "sloth bear",
    "wild boar",
    "wild cat",
    "none",
]

_NAMES4 = [
    "chithal",
    "cows",
    "goat",
    "sambhar",
    "none",
]



# Name of the dataset usually match the script name with CamelCase instead of snake_case
class CameraTraps(datasets.GeneratorBasedBuilder):
    """Camerat Trap dataset for Telangana AI Forest Challenge."""

    VERSION = datasets.Version("1.2.0")

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
        datasets.BuilderConfig(name="tai4species", version=VERSION, description="This part of my dataset covers a first domain with 4 species"),
        datasets.BuilderConfig(name="tai20species", version=VERSION, description="This part of my dataset covers a second domain with 20 species"),
        datasets.BuilderConfig(name="taiALLspecies", version=VERSION, description="This part of my dataset covers a second domain with 20 species + ALL data (no aug)"),
    ]

    DEFAULT_CONFIG_NAME = "tai4species"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "tai4species":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "filename": datasets.Value("string"),
                    "image": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=_NAMES4),
                    # These are the features of your dataset like images, labels ...
                }
            )
        elif self.config.name in ["tai20species", "taiALLspecies"]: 
            features = datasets.Features(
                {
                    "filename": datasets.Value("string"),
                    "image": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=_NAMES20),
                    # These are the features of your dataset like images, labels ...
                }
            )

            
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,# If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
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
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": data_dir,
                    "filepath": os.path.join(data_dir, "cameratraps/cameratraps_train.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": data_dir,
                    "filepath": os.path.join(data_dir, "cameratraps/cameratraps_val.csv"),
                    "split": "val"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, data_dir, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        reader = pd.read_csv(filepath)
        for key, row in reader.iterrows():
            data =dict(row)
                
            # Yields examples as (key, example) tuples
            yield key, {
                "filename": data["filename"],
                "image": Image.open(os.path.join(data_dir, data["filename"])),
                "label": data["labels"],
                }
          
