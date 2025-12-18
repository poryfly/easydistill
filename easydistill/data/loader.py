# Copyright 2025 the LlamaFactory team.
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

import os
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict

import logging
import json
from .converter import align_dataset
from .data_utils import Role, merge_dataset
from .parser import get_dataset_list


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from .data_args import DataArguments, FILEEXT2TYPE
    from .data_utils import DatasetModule
    from .parser import DatasetAttr


logger = logging.getLogger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        raise ValueError(f"Dataset not support for ms_hub.")

    elif dataset_attr.load_from == "om_hub":
        raise ValueError(f"Dataset not support for om_hub.")
    elif dataset_attr.load_from == "cloud_file":
        raise ValueError(f"Dataset not support for cloud_fle.")
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            num_proc=data_args.preprocessing_num_workers,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args)


def _get_merged_dataset(
    dataset_names: Optional[list[str]],
    data_args: "DataArguments",
    return_dict: bool = False,
) -> Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        datasets[dataset_name] = _load_single_dataset(dataset_attr, data_args)

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=data_args.seed)

def get_dataset(
    data_conf: "dict",
) -> "DatasetDict":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    data_args = DataArguments(**data_conf)
    dataset = _get_merged_dataset(data_args.dataset, data_args)
    # just train
    return DatasetDict({"train":dataset})

