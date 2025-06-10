import sys
sys.path.append("..")
from utils.utils import load_raw_dataset

def load_data(
    path,
    pre_context = [], # for example: [{"role": "system", "content": "You are a helpful assistant."}]
):
    dataset = load_raw_dataset(path, is_lines=False, has_line_index=True)
    new_dataset = []
    for sample in dataset:
        context = pre_context + [{
            "role": "user",
            "content": sample["instruction"],
        }]
        new_sample = {
            "meta": "alpaca_eval:{}".format(sample["line_index"]),
            "context": context,
        }
        new_dataset.append(new_sample)

    return new_dataset