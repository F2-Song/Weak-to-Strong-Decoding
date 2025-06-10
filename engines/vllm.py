import sys
sys.path.append("..")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np

class VLLM_Generator:
    def __init__(
        self, 
        model_path, 
        gpu_memory_utilization=0.9, 
        device="cuda",
        stops=None,
        tensor_parallel_size=4,
        pipeline_parallel_size=4,
        enable_prefix_caching=False,
    ):
        self.model = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            device=device,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            enforce_eager=True,
            enable_chunked_prefill=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.stops = [self.tokenizer.eos_token]
        if stops is not None:
            self.stops.extend(stops)

    def generate(
        self,
        contexts=None,
        apply_template=None,
        prompt_token_ids=None,
        num_return_sequences=1,
        temperature=0,
        max_tokens=128,
        top_p=1,
    ):
        sampling_params = SamplingParams(
            n=num_return_sequences,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=self.stops,
            prompt_logprobs=0,
            logprobs=0,
        )
        print(sampling_params)
        if contexts is not None:
            assert apply_template is not None
            assert prompt_token_ids is None
            original_prompt_token_ids = []
            prompt_token_ids = []
            for context in contexts:
                prompt, predicted_content = apply_template(context)
                if predicted_content is None:
                    input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                    original_prompt_token_ids.append(input_ids)
                    prompt_token_ids.append(input_ids)
                else:
                    input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                    predicted_input_ids = self.tokenizer.encode(predicted_content, add_special_tokens=False)
                    original_prompt_token_ids.append(input_ids)
                    prompt_token_ids.append(input_ids + predicted_input_ids)
        else:
            assert prompt_token_ids is not None
            original_prompt_token_ids = prompt_token_ids
        

        outputs = self.model.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        total_results = []
        for output, one_prompt_token_ids, one_original_prompt_token_ids in zip(outputs, prompt_token_ids, original_prompt_token_ids):
            prompt_logprobs = []
            for token_id, token_logprop_dict in zip(one_prompt_token_ids[1:], output.prompt_logprobs[1:]):
                prompt_logprobs.append(token_logprop_dict[token_id].logprob)
            prompt_logprobs = np.array(prompt_logprobs)
            prompt_probs = np.exp(prompt_logprobs).tolist()
            assert len(one_prompt_token_ids) == len(prompt_probs) + 1
            if len(one_prompt_token_ids) > len(one_original_prompt_token_ids):
                original_prompt_probs = prompt_probs[:len(one_original_prompt_token_ids[1:])]
                assert len(one_original_prompt_token_ids) == len(original_prompt_probs) + 1
                predicted_probs = prompt_probs[len(one_original_prompt_token_ids[1:]):]
                predicted_token_ids = one_prompt_token_ids[len(one_original_prompt_token_ids):]
            else:
                original_prompt_probs = prompt_probs
                predicted_probs = []
                predicted_token_ids = []

            output_list = []
            for sub_output in output.outputs:
                token_ids = []
                logprobs = []
                for token_id, token_logprop_dict in zip(sub_output.token_ids, sub_output.logprobs):
                    token_ids.append(token_id)
                    logprobs.append(token_logprop_dict[token_id].logprob)
                logprobs = np.array(logprobs)
                probs = np.exp(logprobs).tolist()

                token_ids = predicted_token_ids + token_ids
                probs = predicted_probs + probs
                assert len(token_ids) == len(probs)
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

                output_list.append(
                    {
                        "text": text,
                        "token_ids": token_ids,
                        "probs": probs,
                        "finish_reason": sub_output.finish_reason,
                    }
                )
            result = {
                "prompt_token_ids": one_original_prompt_token_ids,
                "prompt_probs": original_prompt_probs,
                "outputs": output_list,
            }
            total_results.append(result)
        
        return total_results