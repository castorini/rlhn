import json
import logging
import os

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
    ):
        self.hf_dataset = hf_dataset 
        self.model_name_or_path = model_name_or_path
        self.model_client = self.get_client(
            client_name=client_name,
            model_name_or_path=model_name_or_path
        )
        self.output_dir = output_dir
        self.output_file = output_file

    @staticmethod
    def get_client(client_name: str, model_name_or_path: str = None):
        if client_name == "openai":
            return OpenAIAzureClient(model_name_or_path=model_name_or_path)
        else:
            raise ValueError(f"Client {client_name} is not supported, currently only OpenAI Azure client is supported.")

    @staticmethod
    def format_example(example):
        """Format the example to be used in the prompt."""
        question = example['query']
        positive_passage_text, positive_passage_ids = "", []
        for idx, passage in enumerate(example['positive_passages']):
            positive_passage_text += "P(0): " + passage['text'] + "\n\n"
            positive_passage_ids.append(passage['docid'])

        negative_passage_text, negative_passage_ids = "", []
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

