export OMP_NUM_THREADS=1
export HF_HOME=/mnt/users/n3thakur/cache
export DATASETS_HF_HOME=/mnt/users/n3thakur/cache
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_ENDPOINT="https://trec-rag-2025.openai.azure.com"
export AZURE_OPENAI_API_KEY="3h8RKDbIKVyfGQWwq0TyxWcRRlows6sjX3PuI6VenckqFxD23GJRJQQJ99BAAC4f1cMXJ3w3AAABACOGiFaz"
export DATASETS=("fiqa")

for dataset in "${DATASETS[@]}"; do
    python -m false_negative \
        --model_name_or_path gpt-4o \
        --subset ${dataset} \
        --train_dataset RLHN/bge-retrieval-data-default-100K \
        --output_dir ./output/bge-retrieval-data-default-100K/ \
        --output_file ${dataset}.gpt-4o.jsonl \
        --max_completion_tokens 4096 \
        --temperature 0.1 \
        --prompt_version gpt-4o 
done
