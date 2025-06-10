import sys
sys.path.append("..")
from utils.utils import load_raw_dataset

def load_data(
    path,
    pre_context = [],
):
    dataset = load_raw_dataset(path, is_lines=True, has_line_index=True)
    
    for sample in dataset:
        sample["context"] = pre_context + sample["raw_context"]
        sample["meta"] = "hh_{}".format(sample["line_index"])
    
    return dataset