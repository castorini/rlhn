export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export DATASETS=("nfcorpus" "scifact" "scidocs" "arguana" "fiqa" "trec-covid")
export MODELS=("bge-retrieval-data-stage-2-rlhn-680K")

for model in ${MODELS[@]}; do
    python -m e5_evaluate_beir \
        --model_name_or_path RLHN/e5-base-$model \
        --output_dir results/beir/e5-base-$model \
        --datasets ${DATASETS[@]} \
        --pooling mean \
        --score_function cos_sim \
        --max_length 512 \
        --normalize \
        --query_prompt "query: " \
        --passage_prompt "passage: "
done