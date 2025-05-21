import csv
import json
import logging
import os
import random
import time

random.seed(42)

from datetime import datetime

from datasets import Dataset
from tqdm.autonotebook import tqdm

from rlhn import LoggingHandler
from rlhn.clients import OpenAIAzureClient
from rlhn.prompts import RLHNPrompt

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

class RLHN:
    def __init__(self,
            hf_dataset: Dataset,
            client_name: str = "openai",
            model_name_or_path: str = None,
            output_dir: str = "output",
            output_file: str = "rlhn.jsonl",
            batch_jobs_csv: str = "openai_jobs.csv",
    ):
        self.hf_dataset = hf_dataset 
        self.model_name_or_path = model_name_or_path
        self.client_name = client_name
        self.model_client = self.get_client(
            client_name=client_name,
            model_name_or_path=model_name_or_path
        )
        self.output_dir = output_dir
        self.output_file = output_file
        self.batch_jobs_csv = batch_jobs_csv

    @staticmethod
    def get_client(client_name: str, model_name_or_path: str = None):
        if client_name == "openai":
            return OpenAIAzureClient(model_name_or_path=model_name_or_path)
        else:
            raise ValueError(f"Client {client_name} is not supported, currently only OpenAI Azure client is supported.")

    @staticmethod
    def format_example(example, sample: bool = False, sample_max: int = 25):
        """Format the example to be used in the prompt."""
        question = example['query']
        positive_passage_text, positive_passage_ids = "", []
        for idx, passage in enumerate(example['positive_passages']):
            positive_passage_text += "P(0): " + passage['text'] + "\n\n"
            positive_passage_ids.append(passage['docid'])

        negative_passage_text, negative_passage_ids = "", []
        if sample:
            random.shuffle(example['negative_passages'])
            for idx, passage in enumerate(example['negative_passages'][:sample_max]):
                negative_passage_text += f"Doc ({idx+1}): " + passage['text'] + "\n\n"
                negative_passage_ids.append(passage['docid'])
        else:
            for idx, passage in enumerate(example['negative_passages']):
                negative_passage_text += f"Doc ({idx+1}): " + passage['text'] + "\n\n"
                negative_passage_ids.append(passage['docid'])

        kwargs = {
                "question": question,
                "ground_truth": positive_passage_text.strip(),
                "documents": negative_passage_text.strip()
        }
        return kwargs, positive_passage_ids, negative_passage_ids

    def cost(
        self,
        prompt_cls: RLHNPrompt,
        max_completion_tokens: int = 2048,
        temperature: float = 0.1,
        skip_query_ids: list = [],
        filter_query_ids: list = [],
    ):
        overall_cost = 0.0
        for example in tqdm(
                self.hf_dataset,
                total=len(self.hf_dataset),
                desc=f"Filtering with {self.model_name_or_path} and prompt: {prompt_cls.version}"
            ):
            ### Check if the query id is skip query ids
            if skip_query_ids:
                if example["query_id"] in skip_query_ids:
                    continue

            if filter_query_ids:
                ### Filter the dataset based on the query ids
                if example["query_id"] in filter_query_ids:
                    kwargs, _, _ = self.format_example(example)
                    prompt = prompt_cls.get_prompt(**kwargs)
                    estimated_cost = self.model_client.cost(prompt=prompt, max_tokens=max_completion_tokens)
                    overall_cost += estimated_cost
            else:
                kwargs, _, _ = self.format_example(example)
                prompt = prompt_cls.get_prompt(**kwargs)
                estimated_cost = self.model_client.cost(prompt=prompt, max_tokens=max_completion_tokens)
                overall_cost += estimated_cost

        ### Print the overall cost
        logging.info(f"Estimated cost for the dataset: {overall_cost:.2f} USD for {len(self.hf_dataset)} examples...")

    def call(
            self,
            prompt_cls: RLHNPrompt,
            max_completion_tokens: int = 2048,
            temperature: float = 0.1,
            skip_query_ids: list = [],
        ):
        """Call the prompt on all examples in the dataset and save output as JSONL."""

        ### Save the output to the output directory
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, f"{self.output_file}"), "a", encoding="utf-8") as f:
            for example in tqdm(
                self.hf_dataset,
                total=len(self.hf_dataset),
                desc=f"Filtering with {self.model_name_or_path} and prompt: {prompt_cls.version}"
            ):
                ### Check if the query id is skip query ids
                if skip_query_ids:
                    if example["query_id"] in skip_query_ids:
                        continue

                kwargs, positive_passage_ids, negative_passage_ids = self.format_example(example)
                prompt = prompt_cls.get_prompt(**kwargs)
                output = self.model_client.response(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_completion_tokens,
                    n=1,
                    disable_logging=True,
                )

                ### Save the output to the example
                example['positive_passage_ids'] = positive_passage_ids
                example['negative_passage_ids'] = negative_passage_ids
                example['response'] = output.choices[0].message.content
                example['finish_reason'] = output.choices[0].finish_reason

                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                f.flush()

    def batch_call(
            self,
            prompt_cls: RLHNPrompt,
            max_completion_tokens: int = 2048,
            temperature: float = 0.1,
            filter_query_ids: list = [],
            chunk_size: int = 250,
            start: int = 0,
            end: int = None,
            sample: bool = False,
            sample_max: int = 25,
        ):
        """Call the prompt on all examples in the dataset and save output as JSONL."""

        ### if chunk size > 250, you need to check the batch size
        if chunk_size > 250:
            logging.warning(f"Your chunk size is greater than 250, we advise a maximum chunk size of {chunk_size} to avoid random missing rows. \
                            Check this link for more info: https://learn.microsoft.com/en-us/answers/questions/2225417/azure-openai-batch-job-(gpt-4o-mini)-completes-ear.")

        ### Save the output to the output directory
        os.makedirs(self.output_dir, exist_ok=True)
        prompt_version = prompt_cls.version
        filter_dataset = self.hf_dataset

        if filter_query_ids:
            ### Filter the dataset based on the query ids
            filter_dataset = filter_dataset.filter(lambda example: example['query_id'] in filter_query_ids)
            logging.info(f"Filtered dataset based on query ids: {len(filter_dataset)}")

        end = end if end is not None else len(filter_dataset)
        itr = range(start, end, chunk_size)

        # Output each chunk to a separate file
        for start_file, start in enumerate(itr):

            ### Refresh a new client for every 25 files
            if start_file % 25 == 0 and start_file >= 25:
                self.model_client = self.get_client(
                    client_name=self.client_name,
                    model_name_or_path=self.model_name_or_path
                )

            ### Save the input output to ${start_file}.input.jsonl and batch job to ${start_file}.{output_file}.jsonl
            os.makedirs(os.path.join(self.output_dir, "azure_batches"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "input_batches"), exist_ok=True)

            ### Create the batch job file
            batch_filepath = os.path.join(self.output_dir, "azure_batches", f"{start_file}.{self.output_file}")
            input_filepath = os.path.join(self.output_dir, "input_batches", f"{start_file}.{self.output_file}")

            with open(batch_filepath, "w", encoding="utf-8") as f:
                with open(input_filepath, "w",  encoding="utf-8") as fout:

                    end_idx = start+chunk_size if start+chunk_size < end else end
                    indexes = [i for i in range(start, end_idx)]
                    logging.info(f"Processing chunk: {start} - {start+chunk_size} ({len(indexes)})")
                    filter_dataset_sliced = filter_dataset.select(indexes, keep_in_memory=True)
                    logging.info(f"Starting batch chunk: {batch_filepath}")
                    logging.info(f"Starting input chunk: {input_filepath}")

                    for idy, example in tqdm(enumerate(filter_dataset_sliced), total=len(filter_dataset_sliced), desc="Loading Dataset"):
                        kwargs, positive_passage_ids, negative_passage_ids = self.format_example(
                            example, sample=sample, sample_max=sample_max
                        )
                        prompt = prompt_cls.get_prompt(**kwargs)
                        batch_job = {
                            "custom_id": f"{start_file}_{idy}_{prompt_version}_{example['query_id']}",
                            "method": "POST",
                            "url": "/chat/completions",
                            "body": {
                                "model": self.model_client.deployment_name,
                                "messages": [{"role": "user", "content": f"{prompt}"}],
                                "max_completion_tokens": max_completion_tokens,
                                "temperature": temperature
                            }
                        }
                        example['custom_id'] = batch_job['custom_id']
                        example['positive_passage_ids'] = positive_passage_ids
                        example['negative_passage_ids'] = negative_passage_ids
                        del example["positive_passages"], example["negative_passages"]

                        #### Save the example to the input file
                        fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                        fout.flush()

                        #### Save the batch job to the batch file 
                        f.write(json.dumps(batch_job, ensure_ascii=False) + "\n")
                        f.flush()

            ### Upload the batch job to the OpenAI Azure Batch API
            with open(self.batch_jobs_csv, "a", encoding="utf-8") as f:
                writer = csv.writer(f)

                #### Upload a file with a purpose of "batch"
                file = self.model_client.client.files.create(
                    file=open(batch_filepath, "rb"),
                    purpose="batch"
                )

                logging.info(file.model_dump_json(indent=2))
                file_id = file.id

                logging.info(f"Batch job file uploding: {file_id}")
                ### Submit a batch job with the file
                batch_response = self.model_client.client.batches.create(
                    input_file_id=file_id,
                    endpoint="/chat/completions",
                    completion_window="24h",
                )

                batch_id = batch_response.id
                logging.info(batch_response.model_dump_json(indent=2))

                logging.info(f"Ingested file id: {file_id}")
                logging.info(f"Batch job submitted: {batch_id}")

                status = "validating"
                while status not in ("completed", "failed", "canceled"):
                    time.sleep(60) ## sleep for 1 minute
                    batch_response = self.model_client.client.batches.retrieve(batch_id)
                    status = batch_response.status
                    logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Batch Id: {batch_id},  Status: {status}")

                logging.info(f"Batch job {batch_id} status: {status}")
                writer.writerow([
                    prompt_version,
                    batch_filepath,
                    file_id,
                    batch_id,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    status
                ])
                f.flush()
