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
import ijson
from easydistill.data.data_utils import DataField, Role

def load_dataset_from_json(
    data_path: str) -> Dataset:

    load_data = None
    all_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for item in ijson.items(f, 'item'):
            if DataField.INSTRUCTION in item:
                messages = [
                    {DataField.ROLE: Role.SYSTEM, DataField.CONTENT: item[DataField.INSTRUCTION]},
                    {DataField.ROLE: Role.USER, DataField.CONTENT: item[DataField.INPUT]}
                ]
                if DataField.OUTPUT in item:
                    messages.append({DataField.ROLE: Role.ASSISTANT, DataField.CONTENT: item[DataField.OUTPUT]})
            elif DataField.MESSAGES in item:
                messages = item[DataField.MESSAGES]
            all_data.append({DataField.MESSAGES:messages})

    if len(all_data) > 0:
        load_data = Dataset.from_list(all_data)
    return load_data


