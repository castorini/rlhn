export OMP_NUM_THREADS=1
export HF_HOME=/mnt/users/n3thakur/cache
export DATASETS_HF_HOME=/mnt/users/n3thakur/cache
export AZURE_OPENAI_API_VERSION="xxxx"
export AZURE_OPENAI_ENDPOINT="xxxx"
export AZURE_OPENAI_API_KEY="xxxx"
export DATASETS=("fiqa")

for dataset in "${DATASETS[@]}"; do
    python -m false_negative_batch \
        --model_name_or_path gpt-4o-mini-batch \
        --subset ${dataset} \
        --train_dataset RLHN/bge-retrieval-data-default-100K \
        --output_dir ./output/bge-retrieval-data-default-100K/${dataset}_batch/ \
        --output_file ${dataset}.gpt-4o-batch.jsonl  \
        --max_completion_tokens 4096 \
        --temperature 0.1 \
        --prompt_version gpt-4o-mini \
        --jobs_csv ./output/bge-retrieval-data-default-100K/${dataset}_batch/openai_${dataset}.csv \
        --batch_job \
        --chunk_size 250 \
        --query_ids_file ./test/${dataset}.query_ids.txt
done
