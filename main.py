from utils.config import args
from inference.draft import generate as draft_generate
from inference.check import draft_check
from inference.target import generate as target_generate
from utils.utils import save_dataset
from functools import partial
import yaml

def save_args_to_yaml(args, filename):
    args_dict = vars(args)
    with open(filename, "w") as f:
        yaml.dump(args_dict, f)

if __name__ == '__main__':
    save_args_to_yaml(
        args, 
        args.config_path,
    )

    if args.mode == "draft":
        if "naive" in args.dataset_path:
            from load_data.naive import load_data
        elif "alpaca_eval" in args.dataset_path:
            from load_data.alpaca_eval import load_data
        elif "arena_hard" in args.dataset_path:
            from load_data.arena_hard import load_data
        elif "truthfulQA" in args.dataset_path:
            from load_data.truthfulQA import load_data
        elif "mt_bench" in args.dataset_path:
            from load_data.mt_bench import load_data
            if "logs" not in args.dataset_path:
                # mt_bench_1
                load_data = partial(
                    load_data, 
                    mt_turn=1,
                )
            else:
                # mt_bench_2
                load_data = partial(
                    load_data, 
                    mt_turn=2,
                )
        elif "hh" in args.dataset_path:
            # we are sorry but we do not release the dataset, because it contains offensive content
            # feel free to contact us if you really need it, but its purpose should be on RESEARCH ONLY
            from load_data.hh import load_data
        else:
            raise NotImplementedError
        samples = load_data(args.dataset_path)
        draft_generate(samples, args)
    elif args.mode == "check":
        from load_data.naive import load_data
        samples = load_data(args.dataset_path)
        draft_check(samples, args)
    elif args.mode == "target":
        from load_data.naive import load_data
        samples = load_data(args.dataset_path)
        target_generate(samples, args)
    else:
        raise NotImplementedError

    save_dataset(samples, args.output_path)
