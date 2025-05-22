import json
import logging
import random

random.seed(42)


from datasets import Dataset

from rlhn import LoggingHandler
from rlhn.dataset.techniques import RLHNTechnique

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

class RLHNDataset:
    def __init__(self, technique: str, stage_1_files: list[str] = [], stage_2_files: list[str] = []):
        """
        """
        self.stage_1_files = stage_1_files
        self.stage_2_files = stage_2_files
        self.technique = technique
        self.rlhn_technique = RLHNTechnique()

    def check_arguana(self, dataset_filename: str):
        """
        Check if the dataset is arguana or not
        """
        if "arguana" in dataset_filename:
            return "default"
        else:
            return self.technique

    def process(self, stage: int, filter_string: str, output_hf: str, private: bool):
        output_rows = []

        if stage == 1:
            for dataset_file in self.stage_1_files:
                technique = self.check_arguana(dataset_file)
                logging.info(f"Processing: {dataset_file} with technique: {technique}")
                with open(dataset_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        modified_data = self.rlhn_technique.modify(
                            data,
                            technique=technique,
                            filter_string=filter_string,
                        )
                        if modified_data:
                            output_rows.append(modified_data)

        elif stage == 2:
            for stage_1_file, stage_2_file in zip(self.stage_1_files, self.stage_2_files):
                technique = self.check_arguana(stage_2_file)
                logging.info(f"Processing: {stage_2_file} with technique: {technique}")
                query_ids, query_ids_completed = [], []

                with open(stage_1_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        query_ids.append(data["query_id"])

                with open(stage_2_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if data["query_id"] not in query_ids:
                            continue

                        modified_data = self.rlhn_technique.modify(
                            data,
                            technique=technique,
                            filter_string=filter_string,
                        )

                        if modified_data:
                            output_rows.append(modified_data)
                            query_ids_completed.append(data["query_id"])

                ### For query_ids not completed, filter them with the original dataset
                query_ids = set(query_ids) - set(query_ids_completed)
                logging.info(f"query_ids completed: {len(query_ids_completed)}")
                logging.info(f"query_ids remaining: {len(query_ids)}")

                if len(query_ids) > 0:
                    logging.info(f"Filtering {len(query_ids)} query_ids with the original dataset")
                    with open(stage_1_file, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            modified_data = self.rlhn_technique.modify(
                                data, technique="default", filter_string=filter_string,
                            )
                            if modified_data:
                                output_rows.append(modified_data)

        #### Create a HF dataset from the output list of dictionaries
        logging.info(f"Dataset length: {len(output_rows)}")
        hf_dataset = Dataset.from_list(output_rows)

        ## create a dataset with the same schema as the original dataset
        logging.info(f"Dataset length: {len(hf_dataset)}")
        logging.info(f"Dataset columns: {hf_dataset.column_names}")
        logging.info("first row: ", hf_dataset[0])

        ### upload the dataset to HF
        hf_dataset.push_to_hub(
            output_hf,
            config_name="default",
            private=private,
            split="train"
        )
