import math
import sys
sys.path.append("..")
from engines.hf_encoder import HF_Encoder
from inference.draft import apply_template
from functools import partial

def target_apply_template(
        context,
        drafted_response,
        tokenizer, 
        is_instruct_model=False,
    ):
    if context[-1]["role"] == "assistant":
        # the prefix should have been included in drafted_response, so we can remove the last assistant message here
        context = context[:-1]
    context = context + [{"role": "assistant", "content": drafted_response}]
    origin_prompt, drafted_response = apply_template(
        context=context,
        tokenizer=tokenizer,
        is_instruct_model=is_instruct_model,
        add_generation_prompt=True,
    )
    drafted_prompt = (origin_prompt + drafted_response).rstrip()
    origin_prompt_len = len(tokenizer.encode(origin_prompt, add_special_tokens=False))
    drafted_prompt_ids = tokenizer.encode(drafted_prompt, add_special_tokens=False)
    origin_prompt_ids = drafted_prompt_ids[:origin_prompt_len]
    return origin_prompt_ids, drafted_prompt_ids
    
def get_accept_index(
        probs,
        threshold=0.8,
        window_size=6,
    ):
    avg_probs = []
    for i in range(len(probs)):
        if i+1-window_size < 0:
            avg_probs.append(0)
        else:
            avg_probs.append(math.pow(math.prod(probs[i+1-window_size:i+1]), 1/window_size))

    accept_indices = [i for i, p in enumerate(avg_probs) if p >= threshold]
    if len(accept_indices) == 0:
        accept_index = len(probs) - 1
    else:
        accept_index = accept_indices[0]
    
    return accept_index

def check(
        samples,
        get_accept_index_fn,
        args,
    ):
    encoder = HF_Encoder(
        args.target_model_path,
    )

    total_origin_prompt_ids = []
    total_drafted_prompt_ids = []
    num_logits_to_keep = 0
    for sample in samples:
        for output in sample[sample["result_key"]]["outputs"]:
            origin_prompt_ids, drafted_prompt_ids = target_apply_template(
                context=sample["context"],
                drafted_response=output["text"],
                tokenizer=encoder.tokenizer,
                is_instruct_model=args.is_instruct_model,
            )
            total_origin_prompt_ids.append(origin_prompt_ids)
            total_drafted_prompt_ids.append(drafted_prompt_ids)
            num_logits_to_keep = max(num_logits_to_keep, len(drafted_prompt_ids) - len(origin_prompt_ids) + 10)

    results = encoder.encode(
        prompt_token_ids=total_drafted_prompt_ids,
        num_logits_to_keep=num_logits_to_keep,
    )

    assert len(results) == len(total_origin_prompt_ids)

    accept_indices = []
    total_drafted_part_probs = []
    for origin_prompt_ids, result in zip(total_origin_prompt_ids, results):
        drafted_part_probs = result["multi_layer_prompt_probs"][-1][len(origin_prompt_ids)-1:]
        total_drafted_part_probs.append(drafted_part_probs)

        accept_index = get_accept_index_fn(drafted_part_probs)
        accept_indices.append(accept_index + len(origin_prompt_ids))
    
    total_index = 0
    for sample in samples:
        sample["target_input"] = []
        for output in sample[sample["result_key"]]["outputs"]:
            sample["target_input"].append({
                # original prompt, i.e. the context
                "origin_prompt_token_ids": total_origin_prompt_ids[total_index],
                # drafted prompt, i.e. the context + drafted response
                "drafted_prompt_token_ids": total_drafted_prompt_ids[total_index],
                # accept index, counting from the context
                "accept_index": accept_indices[total_index],
                # partial accept index, counting from the drafted prompt
                "partial_accept_index": accept_indices[total_index] - len(total_origin_prompt_ids[total_index]),
                # accepted drafted part token ids
                "token_ids": total_drafted_prompt_ids[total_index][len(total_origin_prompt_ids[total_index]):accept_indices[total_index]+1],
                # accepted drafted part
                "text": encoder.tokenizer.decode(total_drafted_prompt_ids[total_index][len(total_origin_prompt_ids[total_index]):accept_indices[total_index]+1], skip_special_tokens=False),
                # drafted part probs, including the accepted part but can be longer than the accepted part
                "probs": total_drafted_part_probs[total_index],
            })
            total_index += 1
        sample["result_key"] = "target_input"


def draft_check(
        samples,
        args,
    ):
    partial_get_accept_index = partial(
        get_accept_index,
        threshold=args.target_get_accept_index_threshold,
        window_size=args.target_get_accept_index_window_size,
    )    
    check(
        samples=samples,
        get_accept_index_fn=partial_get_accept_index,
        args=args,
    )