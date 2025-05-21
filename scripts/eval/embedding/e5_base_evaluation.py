from rlhn.dataloader import AirBenchDataLoader

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib
import os
import argparse


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=False)
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--benchmark", type=str, default="beir", choices=["beir", "air-bench"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pooling", type=str, required=True, choices=['mean', 'cls'])
    parser.add_argument("--score_function", type=str, required=True, choices=['cos_sim', 'dot'])
    parser.add_argument("--max_length", type=int, default=350)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--query_prompt", type=str, default=None, required=False)
    parser.add_argument("--passage_prompt", type=str, default=None, required=False)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    #### Load the encoder model: e5-base
    dense_model = models.HuggingFace(
        args.model_name_or_path,
        max_length=args.max_length,
        pooling=args.pooling,
        normalize=args.normalize,
        prompts={
            "query": args.query_prompt, 
            "passage": args.passage_prompt
        },
    )

    #### Download scifact.zip dataset and unzip the dataset
    for dataset in args.datasets:
        if os.path.exists(os.path.join(args.output_dir, f"{dataset}.json")):
            print(f"Skipping {dataset} as it has already been evaluated.")
            continue

        if args.benchmark == "beir":
            #### Load the BEIR dataset
            #### Download ${dataset}.zip dataset and unzip the dataset
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
            data_path = util.download_and_unzip(url, out_dir)

            #### Provide the data_path where scifact has been downloaded and unzipped
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        
        elif args.benchmark == "air-bench":
            #### Load the AIR-Bench dataset
            dataloader = AirBenchDataLoader('AIR-Bench_24.05', cache_dir=args.cache_dir)
            corpus, queries, qrels = dataloader.load(
                task_type='qa', 
                domain=dataset, 
                language='en', 
                dataset_name='default', 
                split="dev"
            )
        
        #### Load the Dense Retriever model
        model = DRES(dense_model, batch_size=args.batch_size)
        retriever = EvaluateRetrieval(model, score_function=args.score_function) # or "cos_sim" for cosine similarity
        results = retriever.retrieve(corpus, queries)

        #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, "mrr")

        #### Save the runfile and the evaluation results
        output_results_dir = args.output_dir
        os.makedirs(output_results_dir, exist_ok=True)
        util.save_runfile(os.path.join(output_results_dir, f"{dataset}.run.trec"), results)

        #### Save the evaluation results
        util.save_results(os.path.join(output_results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)

if __name__ == "__main__":
    main()