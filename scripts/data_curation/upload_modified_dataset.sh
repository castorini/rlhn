# DATASETS=(
#     "<your_saved_output_dir>/250K/msmarco_passage.stage_1.gpt-4o-mini.jsonl" 
#     "<your_saved_output_dir>/250K/fiqa.stage_1.gpt-4o-mini.jsonl" 
#     "<your_saved_output_dir>/250K/fever.stage_1.gpt-4o-mini.jsonl"
#     "<your_saved_output_dir>/250K/arguana.stage_1.gpt-4o-mini.jsonl"
#     "<your_saved_output_dir>/250K/hotpotqa.stage_1.gpt-4o-mini.jsonl"
#     "<your_saved_output_dir>/250K/nq.stage_1.gpt-4o-mini.jsonl"
#     "<your_saved_output_dir>/250K/scidocsrr.stage_1.gpt-4o-mini.jsonl"
# )

python upload_modified_dataset.py \
    --stage_1_files ${DATASETS[@]} \
    --technique rlhn \
    --filter_string "better" \
    --output_hf RLHN/bge-retrieval-data-stage-1-rlhn-250K


# STAGE_1_FILES=(
#     "<your_saved_output_dir>/250K/msmarco_passage.stage_1.gpt-4o-mini.jsonl" 
#     "<your_saved_output_dir>/250K/fiqa.stage_1.gpt-4o-mini.jsonl" 
#     "<your_saved_output_dir>/250K/fever.stage_1.gpt-4o-mini.jsonl"
#     "<your_saved_output_dir>/250K/arguana.stage_1.gpt-4o-mini.jsonl"
#     "<your_saved_output_dir>/250K/hotpotqa.stage_1.gpt-4o-mini.jsonl"
#     "<your_saved_output_dir>/250K/nq.stage_1.gpt-4o-mini.jsonl"
#     "<your_saved_output_dir>/250K/scidocsrr.stage_1.gpt-4o-mini.jsonl"
# )

# STAGE_2_FILES=(
#     "<your_saved_output_dir>/msmarco_passage.stage_2.gpt-4o.jsonl"
#     "<your_saved_output_dir>/fiqa.stage_2.gpt-4o.jsonl" 
#     "<your_saved_output_dir>/fever.stage_2.gpt-4o.jsonl"
#     "<your_saved_output_dir>/arguana.stage_2.gpt-4o.jsonl"
#     "<your_saved_output_dir>/hotpotqa.stage_2.gpt-4o.jsonl"
#     "<your_saved_output_dir>/nq.stage_2.gpt-4o.jsonl"
#     "<your_saved_output_dir>/scidocsrr.stage_2.gpt-4o.jsonl"
# )

python upload_modified_dataset.py \
    --stage_1_files ${STAGE_1_FILES[@]} \
    --stage_2_files ${STAGE_2_FILES[@]} \
    --technique "rlhn" \
    --filter_string "both" \
    --output_hf RLHN/bge-retrieval-data-stage-2-rlhn-250K