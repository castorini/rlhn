"""
Identify and save query_ids from the result file that are identified as false negatives.

Usage:
    export RESULTS_FILEPATH="./test/bge-retrieval-data-default-test-50/msmarco_passage.stage_1.gpt-4o-mini.jsonl"
    export OUTPUT_FILE="./test/bge-retrieval-data-default-test-50/query_ids/msmarco_passage.stage_1.query_ids.with.false_negatives.txt"
    python identify_false_negatives.py --result_filepath $RESULTS_FILEPATH --filter_string "both" --output_filepath $OUTPUT_FILE

"""

import argparse
import json
import logging

from rlhn import LoggingHandler
from rlhn.dataset.techniques import RLHNTechnique

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_filepath", type=str, required=True)
    parser.add_argument("--filter_string", type=str, required=False, default="both")
    parser.add_argument("--output_filepath", type=str, required=True)
    args = parser.parse_args()

    rlhn_technique = RLHNTechnique()
    query_ids = []

    # Check if the result file exists
    with open(args.result_filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            present = rlhn_technique.identify(
                data, filter_string=args.filter_string,
            )
            if present:
                query_ids.append(data["query_id"])

    logging.info(f"Identified query ids: {len(query_ids)}")
    with open(args.output_filepath, "w", encoding="utf-8") as out_f:
        for query_id in query_ids:
            out_f.write(query_id + "\n")

if __name__ == "__main__":
    main()