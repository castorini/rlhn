# The script is used to LoRA finetune the Qwen2.5-7B LLM as an embedding model using tevatron.
# It uses flash-attention for distributed training and saves the model checkpoints.
# We fine-tuned the Qwen2.5-7B model using 2 H200 GPUs with a max of 141GB VRAM each.
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1

# Add the wandb project name, where the training results will be logged
export WANDB_PROJECT=rlhn

# Fine-tuning two rlhn curated training datasets with different sizes: 400K and 680K
# 400K: https://huggingface.co/datasets/RLHN/bge-retrieval-data-stage-2-rlhn-400K (Tevatron Format)
# 680K: https://huggingface.co/datasets/RLHN/bge-retrieval-data-stage-2-rlhn-680K (Tevatron Format)
DATASETS=("bge-retrieval-data-stage-2-rlhn-400K" "bge-retrieval-data-stage-2-rlhn-680K")

# Loop through each dataset and run the training script for each split
for dataset in ${DATASETS[@]}; do
    # Use deepspeed to run the training script with the specified parameters
    # --include specifies the GPUs to use, --master_port specifies the port for distributed training
    # Make sure the GPUs are available and similar to the ones above.
    deepspeed --include localhost:0,1 --master_port 60000 --module tevatron.retriever.driver.train \
        --deepspeed ds_zero0_config.json \
        --output_dir models/Qwen2.5-7B-$dataset \
        --run_name Qwen2.5-7B-$dataset-bsize-64x4x8-epochs-1 \
        --lora \
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
        --model_name_or_path Qwen/Qwen2.5-7B \
        --attn_implementation flash_attention_2 \
        --dtype bfloat16 \
        --save_steps 800 \
        --dataset_name RLHN/$dataset \
        --query_prefix "query: " \
        --passage_prefix "passage: " \
        --bf16 \
        --pooling eos \
        --append_eos_token \
        --normalize \
        --temperature 0.01 \
        --per_device_train_batch_size 64 \
        --gradient_checkpointing \
        --train_group_size 8 \
        --learning_rate 5e-6 \
        --query_max_len 350 \
        --passage_max_len 350 \
        --num_train_epochs 1 \
        --logging_steps 5 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 4
done

### The same script you can set CACHE_DIR and DATASETS_CACHE_DIR to cache the training dataset/models etc.
# export CACHE_DIR=<your_cache_dir>
# export DATASETS_CACHE_DIR=<your_cache_dir>
# deepspeed --include localhost:0,1 --master_port 60000 --module tevatron.retriever.driver.train \
#     --deepspeed ds_zero0_config.json \
#     --output_dir models/Qwen2.5-7B-bge-retrieval-data-stage-2-rlhn-400K \
#     --run_name Qwen2.5-7B-bge-retrieval-data-stage-2-rlhn-400K-bsize-64x4x8-epochs-1 \
#     --lora \
#     --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
#     --model_name_or_path Qwen/Qwen2.5-7B \
#     --cache_dir <your_cache_dir> \
#     --dataset_cache_dir <your_cache_dir> \
#     --attn_implementation flash_attention_2 \
#     --dtype bfloat16 \
#     --save_steps 800 \
#     --dataset_name RLHN/bge-retrieval-data-stage-2-rlhn-400K \
#     --query_prefix "query: " \
#     --passage_prefix "passage: " \
#     --bf16 \
#     --pooling eos \
#     --append_eos_token \
#     --normalize \
#     --temperature 0.01 \
#     --per_device_train_batch_size 64 \
#     --gradient_checkpointing \
#     --train_group_size 8 \
#     --learning_rate 5e-6 \
#     --query_max_len 350 \
#     --passage_max_len 350 \
#     --num_train_epochs 1 \
#     --logging_steps 5 \
#     --overwrite_output_dir \
#     --gradient_accumulation_steps 4
