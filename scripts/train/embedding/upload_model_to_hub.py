"""
This script is used to push a fine-tuned model to Hugging Face Hub. This is how to use it:
<Example usage for Qwen2.5-7B model>

export CUDA_VISIBLE_DEVICES=-1
export HF_HOME=<path_to_your_cache>
export DATASETS_HF_HOME=<path_to_your_cache>

python -m upload_model_to_hub \
    --model_name_or_path models/Qwen2.5-7B-bge-retrieval-data-stage-2-rlhn-400K \
    --base_model_name_or_path Qwen/Qwen2.5-7B \
    --output_hf_repo RLHN/Qwen2.5-7B-bge-retrieval-data-stage-2-rlhn-400K \
    --lora \
    --private \
    --pooling eos \
    --normalize \
    --query_prompt "query: " \
    --passage_prompt "passage: "

<Example usage for e5-base model>

export CUDA_VISIBLE_DEVICES=-1

python -m upload_model_to_hub \
    --model_name_or_path models/e5-base-bge-retrieval-data-stage-2-rlhn-400K \
    --output_hf_repo RLHN/e5-base-bge-retrieval-data-stage-2-rlhn-400K \
    --private \
    --pooling mean \
    --normalize \
    --query_prompt "query: " \
    --passage_prompt "passage: "
"""
# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from rlhn.util import dot_score, mean_pooling, eos_pooling

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output_hf_repo", type=str, required=True)
    parser.add_argument("--pooling", type=str, required=True, choices=['mean', 'cls', 'eos'])
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--lora", action='store_true', help="Whether to use LoRA model")
    parser.add_argument("--query_prompt", type=str, default=None, required=False)
    parser.add_argument("--passage_prompt", type=str, default=None, required=False)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--private", action='store_true', help="Whether to make the model private or public on Hugging Face Hub")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    
    ### Load the model using LoRA with LLMs
    if args.lora:
        base_model = AutoModel.from_pretrained(args.base_model_name_or_path, cache_dir=args.cache_dir)
        model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
        encoder = model.merge_and_unload()
        print("Encoder: ", encoder)
    # if not using LoRA (e.g., using E5-base or encoder models)
    else:
        encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype="auto").to(device)
        print("Encoder: ", encoder)

    # We use msmarco query and passages as an example
    query =  "When was Marie Curie born?"
    contexts = [
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    ]

    if args.query_prompt and args.passage_prompt:
        query = args.query_prompt + query
        contexts = [args.passage_prompt + ctx for ctx in contexts]

    # Apply tokenizer
    query_input = tokenizer(query, return_tensors='pt').to(device)
    ctx_input = tokenizer(contexts, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute embeddings: take the normalized mean pooling of the embeddings
    # Compute token embeddings
    with torch.no_grad():
        # Compute embeddings: take the last-layer hidden state of the [CLS] token
        if args.pooling in ['cls']:
            query_emb = encoder(**query_input).last_hidden_state[:, 0, :]
            ctx_emb = encoder(**ctx_input).last_hidden_state[:, 0, :]
        
        elif args.pooling in ['mean']:
            query_output = encoder(**query_input)
            ctx_output = encoder(**ctx_input)
            query_emb = mean_pooling(query_output, query_input['attention_mask'])
            ctx_emb = mean_pooling(ctx_output, ctx_input['attention_mask'])
        
        elif args.pooling in ['eos']:
            query_output = encoder(**query_input)
            ctx_output = encoder(**ctx_input)
            query_emb = eos_pooling(query_output, query_input['attention_mask'])
            ctx_emb = eos_pooling(ctx_output, ctx_input['attention_mask'])

        
    # if normalization is needed
    if args.normalize:
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        ctx_emb = torch.nn.functional.normalize(ctx_emb, p=2, dim=1)

    # Print the shapes of the embeddings
    print("Shapes of the embeddings: ", query_emb.shape, ctx_emb.shape)

    # Compute similarity scores using dot product
    score1 = dot_score(query_emb, ctx_emb[0])  
    score2 = dot_score(query_emb, ctx_emb[1]) 
    
    # Print the similarity scores
    print("query: ", query)
    print("Documents:", contexts)
    print("Similarity scores: ", score1, score2)

    # Save the model to hub
    if args.lora:
        model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
        model.push_to_hub(repo_id=args.output_hf_repo, private=args.private)
    else:
        encoder.push_to_hub(repo_id=args.output_hf_repo, private=args.private)

    # save the tokenizer to hub
    tokenizer.push_to_hub(repo_id=args.output_hf_repo, private=args.private)


if __name__ == "__main__":
    main()
