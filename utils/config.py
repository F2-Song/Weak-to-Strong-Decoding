import argparse
import sys
sys.path.append("..")
from utils.utils import setup_seed

def parse_args():
    parser = argparse.ArgumentParser(description="weak-to-strong-decoding")
    parser.add_argument(
        "--id", 
        type=str, 
        default="exp_0"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="draft" # draft, target, check
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default=""
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--is_instruct_model",
        action="store_true",
    )
    parser.add_argument(
        "--draft_model_path", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--draft_max_tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--draft_num_return_sequences",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--target_model_path", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--target_max_tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--target_num_return_sequences",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--target_get_accept_index_threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--target_get_accept_index_window_size",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
    )
    args = parser.parse_args()
    return args

args = parse_args()
setup_seed(args.seed)
args_message = '\n'+'\n'.join([f'{k:<40}: {v}' for k, v in vars(args).items()])
print(args_message)