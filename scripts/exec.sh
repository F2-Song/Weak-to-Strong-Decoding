# This script is used to run WSD inference pipeline
# Besides the following arguments, you can also specify others according to utils/config.py

id=$1
test_data_path=$2
draft_model_path=$3
draft_max_tokens=$4
target_model_path=$5
target_max_tokens=$5

mkdir -p logs/$id/inference_res

# draft inference
python -u main.py \
    --id $id \
    --mode draft \
    --tensor_parallel_size 1 \
    --dataset_path $test_data_path \
    --output_path logs/$id/inference_res/draft.json \
    --config_path logs/$id/draft_config.yaml \
    --is_instruct_model \
    --draft_model_path $draft_model_path \
    --draft_max_tokens $draft_max_tokens > logs/$id/draft.log 2>&1

# acquire the position of model switch
python -u main.py \
    --id $id \
    --mode check \
    --dataset_path logs/$id/inference_res/draft.json \
    --output_path logs/$id/inference_res/check.json \
    --config_path logs/$id/check_config.yaml \
    --target_model_path $target_model_path > logs/$id/check.log 2>&1

# target inference
python -u main.py \
    --id $id \
    --mode target \
    --dataset_path logs/$id/inference_res/check.json \
    --output_path logs/$id/inference_res/target.json \
    --config_path logs/$id/target_config.yaml \
    --target_model_path $target_model_path \
    --target_max_tokens $target_max_tokens > logs/$id/target.log 2>&1