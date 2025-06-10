import tqdm
import sys
sys.path.append("..")
from engines.vllm import VLLM_Generator
from functools import partial

def apply_template(
        context, 
        tokenizer, 
        is_instruct_model=False,
        add_generation_prompt=True,
    ):
    predicted_content = None
    if context[-1]["role"] == "assistant":
        predicted_content = context[-1]["content"].strip()
        context = context[:-1]
        
    if is_instruct_model:
        prompt = tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        prompt = ""
        for message_index, message in enumerate(context):
            if message_index == len(context) - 1:
                assert message["role"] == "user"

            if message["role"] == "system":
                prompt += "\n\n[System]\n" + message["content"].strip()
            elif message["role"] == "user":
                prompt += "\n\n[User]\n" + message["content"].strip()
            elif message["role"] == "assistant":
                prompt += "\n\n[Assistant]\n" + message["content"].strip()
        if add_generation_prompt:
            prompt += "\n\n[Assistant]\n"

    if not tokenizer.bos_token is None and not prompt.startswith(tokenizer.bos_token):
        prompt = f"{tokenizer.bos_token}{prompt}"

    return prompt, predicted_content

def generate(
        samples,
        args,
    ):
    if args.is_instruct_model:
        stops = ["<end_of_turn>", "<|eot_id|>", "<|im_end|>"]
    else:
        stops = ["\n\n[User]", "\n\n[System]", "\n\n[Assistant]", "\n[User]", "\n[System]", "\n[Assistant]", "[User]", "[System]", "[Assistant]"]
    generator = VLLM_Generator(
        args.draft_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        device="cuda",
        stops=stops,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )
    partial_apply_template = partial(
        apply_template,
        tokenizer=generator.tokenizer,
        is_instruct_model=args.is_instruct_model,
        add_generation_prompt=True,
    )
    
    contexts = [sample["context"] for sample in samples]

    results = generator.generate(
        contexts=contexts,
        apply_template=partial_apply_template,
        prompt_token_ids=None,
        num_return_sequences=args.draft_num_return_sequences,
        max_tokens=args.draft_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for sample, result in zip(samples, results):
        sample["result_key"] = "draft_result"
        sample["draft_result"] = result