# The script is used to finetune the E5 base model using tevatron.
# It uses distributed training for training and saves the model checkpoints.
# We fine-tuned the E5 base model using 4 A6000 GPUs with a max of 48GB VRAM each.
export CUDA_VISIBLE_DEVICES=0,1,2,3
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
    deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train \
        --deepspeed ds_zero0_config.json \
        --output_dir models/e5-base-$dataset \
        --run_name e5-base-$dataset-bsize-128x4x8-epochs-5 \
        --model_name_or_path intfloat/e5-base-unsupervised \
        --save_steps 500 \
        --dataset_name RLHN/$dataset \
        --attn_implementation eager \
        --query_prefix "query: " \
        --passage_prefix "passage: " \
        --bf16 \
        --pooling mean \
        --normalize \
        --temperature 0.01 \
        --per_device_train_batch_size 128 \
        --gradient_checkpointing \
        --train_group_size 8 \
        --learning_rate 2e-5 \
        --query_max_len 350 \
        --passage_max_len 350 \
        --num_train_epochs 5 \
        --logging_steps 5 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 4
done

### The same script you can set CACHE_DIR and DATASETS_CACHE_DIR to cache the training dataset/models etc.
# export CACHE_DIR=<your_cache_dir>
# export DATASETS_CACHE_DIR=<your_cache_dir>
#
# deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train \
#     --deepspeed ds_zero0_config.json \
#     --output_dir models/e5-base-bge-retrieval-data-stage-2-rlhn-680K \
#     --run_name e5-base-bge-retrieval-data-stage-2-rlhn-680K-bsize-128x4x8-epochs-5 \
#     --model_name_or_path intfloat/e5-base-unsupervised \
#     --cache_dir <your_cache_dir> \
#     --dataset_cache_dir <your_cache_dir> \
#     --save_steps 500 \
#     --dataset_name RLHN/bge-retrieval-data-stage-2-rlhn-680K \
#     --attn_implementation eager \
#     --query_prefix "query: " \
#     --passage_prefix "passage: " \
#     --bf16 \
#     --pooling mean \
#     --normalize \
#     --temperature 0.01 \
#     --per_device_train_batch_size 128 \
#     --gradient_checkpointing \
#     --train_group_size 8 \
#     --learning_rate 2e-5 \
#     --query_max_len 350 \
#     --passage_max_len 350 \
#     --num_train_epochs 5 \
#     --logging_steps 5 \
#     --overwrite_output_dir \
#     --gradient_accumulation_steps 4
