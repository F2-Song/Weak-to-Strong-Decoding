import random
import json

def setup_seed(seed):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def load_raw_dataset(path, is_lines=True, has_line_index=False):
    with open(path, 'r', encoding="utf-8") as f:
        if is_lines:
            dataset = [json.loads(line) for line in f.readlines()]
        else:
            dataset = json.load(f)
    if has_line_index:
        for index, sample in enumerate(dataset):
            sample["line_index"] = index
    return dataset

def save_dataset(dataset, path, flag="w"):
    with open(path, flag, encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False)+"\n")