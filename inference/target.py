import sys
sys.path.append("..")
from engines.vllm import VLLM_Generator

def generate(
        samples,
        args,
    ):
    if args.is_instruct_model:
        stops = ["<end_of_turn>", "<|eot_id|>", "<|im_end|>"]
    else:
        stops = ["\n\n[User]", "\n\n[System]", "\n\n[Assistant]", "\n[User]", "\n[System]", "\n[Assistant]", "[User]", "[System]", "[Assistant]"]
    
    generator = VLLM_Generator(
        args.target_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        device="cuda",
        stops=stops,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )

    # try to control the rest length of the target input while avoiding heavily affecting efficiency
    packed_samples = {}
    margin = 50
    for sample_index in range(len(samples)):
        rest_len = args.target_max_tokens - samples[sample_index]["target_input"][0]["partial_accept_index"] - 1
        
        if rest_len % margin == 0:
            tgt_rest_len = int(rest_len / margin)
        else:
            tgt_rest_len = int(rest_len / margin) + 1
        tgt_rest_len = tgt_rest_len * margin

        if tgt_rest_len <= 0:
            tgt_rest_len = 1
        if tgt_rest_len not in packed_samples:
            packed_samples[tgt_rest_len] = []
        
        packed_samples[tgt_rest_len].append(sample_index)
    
    print({k: len(v) for k, v in packed_samples.items()})

    results = [None for _ in range(len(samples))]
    for rest_len in packed_samples:
        print("Processing samples with rest_len", rest_len)
        partial_samples = [samples[sample_index] for sample_index in packed_samples[rest_len]]
        prompt_token_ids = []
        for sample in partial_samples:
            for target_input in sample["target_input"]:
                prompt_token_ids.append(
                    target_input["drafted_prompt_token_ids"][:target_input["accept_index"]+1]
                )
        partial_results = generator.generate(
            prompt_token_ids=prompt_token_ids,
            num_return_sequences=args.target_num_return_sequences,
            max_tokens=rest_len,
            temperature=0.0,
        )
        for index, sample_index in enumerate(packed_samples[rest_len]):
            results[sample_index] = partial_results[index]

    total_index = 0
    for sample in samples:
        sample["target_result"] = {
            "outputs": [],
        } # the size of outputs should be equal to draft_num_return_sequences * target_num_return_sequences
        for target_input in sample["target_input"]:
            target_result = results[total_index]
            draft_token_ids = target_input["drafted_prompt_token_ids"][len(target_input["origin_prompt_token_ids"]):target_input["accept_index"]+1]
            assert target_result["prompt_token_ids"][len(target_input["origin_prompt_token_ids"]):] == draft_token_ids
            draft_token_probs = target_result["prompt_probs"][len(target_input["origin_prompt_token_ids"])-1:]
            
            assert len(target_result["outputs"]) == args.target_num_return_sequences
            for sub_output in target_result["outputs"]:
                completed_token_ids = draft_token_ids + sub_output["token_ids"]
                completed_probs = draft_token_probs + sub_output["probs"]
                completed_text = generator.tokenizer.decode(completed_token_ids, skip_special_tokens=True)

                # have to truncate the text once again, because the stop tokens may appear in the token ids returned by vLLM
                for stop in stops:
                    if stop in completed_text:
                        completed_text = completed_text[:completed_text.index(stop)]

                sample["target_result"]["outputs"].append({
                    "text": completed_text,
                    "token_ids": completed_token_ids,
                    "probs": completed_probs,
                })
            
            total_index += 1
        sample["result_key"] = "target_result"