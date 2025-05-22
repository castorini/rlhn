"""
"""

from rlhn import LoggingHandler
from rlhn.dataset.upload import RLHNDataset


import logging
import json
import argparse
import re

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_1_files", type=str, required=True, nargs="+")
    parser.add_argument("--stage_2_files", type=str, required=False, nargs="+", default=[])
    parser.add_argument("--technique", type=str, required=True)
    parser.add_argument("--filter_string", type=str, required=False, default="default")
    parser.add_argument("--output_hf", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    rlhn_dataset = RLHNDataset(
        technique=args.technique,
        stage_1_files=args.stage_1_files,
        stage_2_files=args.stage_2_files,
    )

    if len(args.stage_2_files) > 0:
        rlhn_dataset.process(
            stage=2,
            filter_string=args.filter_string,
            output_hf=args.output_hf,
            private=args.private,
        )
    else:
        rlhn_dataset.process(
            stage=1,
            filter_string=args.filter_string,
            output_hf=args.output_hf,
            private=args.private,
        )

if __name__ == "__main__":
    main()