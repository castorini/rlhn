from rlhn import LoggingHandler
from rlhn.prompts import RLHNPrompt
from rlhn.dataset.filtering import RLHN
from datasets import load_dataset
import pyarrow.dataset as pds
import pyarrow.compute as pc

import random
random.seed(42)


import logging
import os, json
import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_completion_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--prompt_version", type=str, default="v1")
    args = parser.parse_args()

    ### Load the filtered dataset query and positive passages as corpus
    hf_dataset = load_dataset(args.train_dataset, split="train")
    logging.info(f"Loading the train dataset ({args.train_dataset})): {len(hf_dataset)}")

    if args.subset:
        ### Filter the dataset based on the subset
        expr = pc.field("subset") == args.subset
        #### https://stackoverflow.com/questions/69290604/pyarrow-table-filtering-huggingface/69296872#69296872
        #### Filtering large HF dataset using pyarrow
        hf_dataset = hf_dataset.with_format("arrow").filter(lambda t: pds.dataset(t).to_table(columns={"mask": expr})[0].to_numpy(),batched=True,).with_format(None)
        logging.info(f"Filtered: train dataset ({args.train_dataset})) & parsed subset: {args.subset}: {len(hf_dataset)}")

    ### load the RLHN prompt along with the dataset
    prompt_cls = RLHNPrompt(version=args.prompt_version)
    logging.info(f"Using prompt: {prompt_cls.__class__.__name__}, version: {prompt_cls.version}")
    logging.info(f"Prompt Template:\n{prompt_cls.template}\n")

    ### create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    ### Check if existing output file exists with completed query_ids
    query_ids_finished = []
    output_filepath = os.path.join(args.output_dir, f"{args.output_file}")
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                if "query_id" in example:
                    query_ids_finished.append(example['query_id'])

    logging.info(f"Already finished query ids: {len(query_ids_finished)}")
    rlhn_class = RLHN(
        hf_dataset=hf_dataset,
        client_name="openai",
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        output_file=args.output_file,
        batch_jobs_csv=args.jobs_csv,
    )

    ### Check the cost of the dataset
    logging.info("Calculating the RLHN filtering the dataset...")
    rlhn_class.cost(
        prompt_cls=prompt_cls,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        skip_query_ids=query_ids_finished,
    )

    # Call one by one of the dataset
    rlhn_class.call(
        prompt_cls=prompt_cls,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        skip_query_ids=query_ids_finished,
    )


if __name__ == "__main__":
    main()