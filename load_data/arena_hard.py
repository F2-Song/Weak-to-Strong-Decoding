import sys
sys.path.append("..")
from utils.utils import load_raw_dataset

def load_data(
    path,
    pre_context = [], # for example: [{"role": "system", "content": "You are a helpful assistant."}]
):
    dataset = load_raw_dataset(path, is_lines=True, has_line_index=True)
    new_dataset = []
    for sample in dataset:
        context = pre_context + [{
            "role": "user",
            "content": sample["turns"][0]["content"],
        }]
        new_sample = {
            "meta": "arena_hard:{}".format(sample["line_index"]),
            "question_id": sample["question_id"],
            "context": context,
        }
        new_dataset.append(new_sample)

    return new_dataset