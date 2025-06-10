import sys
sys.path.append("..")
from utils.utils import load_raw_dataset

def load_data(
    path, # for mt_turn 0, it is the path to the mt_bench dataset; for mt_turn 1, it is the path to the output of llm for mt_turn 0
    pre_context = [], # for example: [{"role": "system", "content": "You are a helpful assistant."}]
    mt_turn = 1, # 1 or 2
):
    if mt_turn == 1:
        dataset = load_raw_dataset(path, is_lines=True, has_line_index=True)
    elif mt_turn == 2:
        dataset = load_raw_dataset(path, is_lines=True, has_line_index=False)
    
    new_dataset = []

    if mt_turn == 1:
        for sample in dataset:
            context = pre_context + [{
                "role": "user",
                "content": sample["turns"][0],
            }]
            new_sample = {
                "meta": "mt_bench_1:{}".format(sample["line_index"]),
                "turns": sample["turns"],
                "question_id": sample["question_id"],
                "context": context,
            }
            new_dataset.append(new_sample)
    elif mt_turn == 2:
        for sample in dataset:
            context = pre_context + [
                {
                    "role": "user",
                    "content": sample["turns"][0],
                },
                {
                    "role": "assistant",
                    "content": sample[sample["result_key"]]["outputs"][0]["text"].strip(),
                },
                {
                    "role": "user",
                    "content": sample["turns"][1],
                }
            ]

            new_sample = {
                "meta": "mt_bench_2:{}".format(sample["meta"].replace("mt_bench_1:", "")),
                "turns": sample["turns"],
                "question_id": sample["question_id"],
                "context": context,
            }

            new_dataset.append(new_sample)
    else:
        raise ValueError(f"mt_turn {mt_turn} not supported; must be 1 or 2")

    return new_dataset