
# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
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
# ==============================================================================

import json, jsonlines
import argparse
import torch
import logging
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from openai import OpenAI
import math
from easydistill.data.loader import load_dataset_from_json
from torch.utils.data import DataLoader
from easydistill.data.data_utils import write_data_to_json_file, Role, DataField


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_json_field(filename, field_name='instruction'):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        output_fields = []
        for item in data:
            if field_name in item:
                output_fields.append(item[field_name])
        return output_fields
    except FileNotFoundError:
        logging.error("The file was not found.")
    except json.JSONDecodeError:
        logging.error("There was an error decoding the JSON file.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")



def load_tokenizer_and_vllm(config, eos_token=None):
    teacher_model_path = config["models"]["teacher"]
    logging.info(f"Loading ckpt and tokenizer: {teacher_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if eos_token:
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        logging.info(f"eos_token {eos_token} from user input")
    elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
        logging.info(f"Initial eos_token_id {tokenizer.eos_token_id} from tokenizer")
        eos_token_id = tokenizer.eos_token_id
        eos_token = tokenizer.convert_ids_to_tokens(eos_token_id)
    else:
        raise ValueError("No available eos_token or eos_token_id.")
    try:
        tokenizer.eos_token = eos_token
        tokenizer.eos_token_id = eos_token_id
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id
    except:
        logging.info(f"[WARNING] Cannot set tokenizer.eos_token")
    logging.info(f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    logging.info(f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")
    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=teacher_model_path,
        tensor_parallel_size=num_gpus,
        enable_chunked_prefill=config["inference"]["enable_chunked_prefill"],
        gpu_memory_utilization=config["inference"]["gpu_memory_utilization"],
        trust_remote_code=config["inference"]["trust_remote_code"],
        dtype=torch.bfloat16,
        enforce_eager=config["inference"]["enforce_eager"],
        max_model_len=config["inference"]["max_model_len"],
    )
    logging.info("vLLM model loaded successfully")
    return tokenizer, llm


def build_template_text(tokenizer, examples):
    messages = examples[DataField.MESSAGES]
    last_message = messages[-1]
    if last_message[DataField.ROLE] == Role.ASSISTANT.value:
        messages = messages[:-1]
    else:
        messages = messages
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return full_text


def generate_teacher_response_batch(tokenizer, llm, data_set, config, batch_size=32):
    outcomes = []
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    for batch in tqdm(dataloader, desc="Generating responses"):
        new_batch = []
        for sample in batch:
            new_batch.append(build_template_text(tokenizer, sample))
        if len(new_batch) == 0:
            continue
        outputs = llm.generate(
            new_batch,
            SamplingParams(
                n = 1,
                top_k = -1,
                temperature = config["inference"]["temperature"],
                seed = config["inference"]["seed"],
                skip_special_tokens = True,
                ignore_eos = False,
                max_tokens = config["inference"]["max_new_tokens"]
            )
        )
        responses = [output.outputs[0].text for output in outputs]
        gen_data = [{DataField.MESSAGES: batch[i][DataField.MESSAGES].append({DataField.ROLE: Role.ASSISTANT.value, DataField.CONTENT: responses[i]})} for i in range(len(batch))]
        outcomes = outcomes + gen_data
    write_data_to_json_file(outcomes, config["data"]["infer_stage_output"])


def generate_teacher_logits_batch(tokenizer, llm, data_set, config, batch_size=32):
    # collate_fn配置很关键，否则字典合并了
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    all_logits = []
    for batch in tqdm(dataloader, desc="Generating responses"):
        new_batch = []
        for sample in batch:
            new_batch.append(build_template_text(tokenizer, sample))
        
        outputs = llm.generate(
            new_batch,  # Pass the raw text directly
            SamplingParams(
                n=1,
                top_k=-1,
                temperature=config["inference"]["temperature"],
                seed=config["inference"]["seed"],
                skip_special_tokens=True,
                ignore_eos=False,
                max_tokens=config["inference"]["max_new_tokens"],
                logprobs=config["inference"]["top_logits_num"],
            )
        )
        # Extract the generated logits
        logits=[output.outputs[0].logprobs for output in outputs]
        for logit in logits:
            for pos in logit:
                for k,v in pos.items():
                    pos[k]=math.exp(v.logprob)
        all_logits = all_logits + logits
    with jsonlines.open(config["data"]["infer_stage_output"], mode='w') as writer:
        for row in all_logits:
            writer.write(row)


def generate_teacher_response_api(data_set, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"]
    )
    models = client.models.list()
    model = models.data[0].id
    logging.info(model)
    stream = config["inference"]["stream"]
    outcomes = []
    for sample in tqdm(data_set, desc="Call remote model and generating responses"):
        messages = sample[DataField.MESSAGES]
        completion = client.chat.completions.create(
            messages = messages,
            model = model,
            max_completion_tokens = config["inference"]["max_new_tokens"],
            stream = stream
        )
        if stream:
            result = ""
            for chunk in completion:
                result += chunk.choices[0].delta.content
        else:
            result = completion.choices[0].message.content

        messages.append({{DataField.ROLE: Role.ASSISTANT.value, DataField.CONTENT: result}})
        outcomes.append({DataField.MESSAGES: messages})
    write_data_to_json_file(outcomes, config["data"]["infer_stage_output"])


def infer_with_teacher_model(config):
    logging.info('Generating distillation data from the teacher model!')
    data_set = load_dataset_from_json(config["data"]["train_data_path"])
    try:
        job_type =  config["job_type"]
        if job_type == "kd_black_box_api":
            generate_teacher_response_api(data_set, config)
        elif job_type == "kd_black_box_local":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            generate_teacher_response_batch(tokenizer, llm, data_set, config)
        elif job_type == "kd_white_box":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            generate_teacher_logits_batch(tokenizer, llm, data_set, config)
        else:
            logging.error(f"Invalid job type: {job_type}")
            raise ValueError(f"Invalid job type: {job_type}")
    except ValueError as e:
        logging.error(f"Training job terminated: {e}")
        return

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    infer_with_teacher_model(config)


if __name__ == "__main__":
    main()