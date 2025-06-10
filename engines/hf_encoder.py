import torch
import torch.nn.functional as F
import sys
sys.path.append("..")

from transformers import AutoTokenizer, AutoConfig
from customized_models.hf_llama import LlamaForMultipleLayerLogprob
from customized_models.hf_gemma2 import Gemma2ForMultipleLayerLogprob
from tqdm import tqdm

MAP_FROM_LM_TP_ENCODER = {
    "LlamaForCausalLM": LlamaForMultipleLayerLogprob,
    "Gemma2ForCausalLM": Gemma2ForMultipleLayerLogprob,
}

class HF_Encoder:
    def __init__(
        self, 
        model_path, 
    ):
        config = AutoConfig.from_pretrained(model_path)

        model = MAP_FROM_LM_TP_ENCODER[config.architectures[0]].from_pretrained(
            model_path,
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        self.model = model
        self.model.gradient_checkpointing_enable()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        torch.set_float32_matmul_precision('high')
    
    def _collate(
            self, 
            batch_prompt_token_ids,
            device=None
            ):
        max_length = max([len(prompt_token_ids) for prompt_token_ids in batch_prompt_token_ids])
        input_ids = []
        attention_mask = []
        for prompt_token_ids in batch_prompt_token_ids:
            input_ids.append(
                prompt_token_ids + [self.tokenizer.pad_token_id] * (max_length - len(prompt_token_ids))
            )
            attention_mask.append(
                [1] * len(prompt_token_ids) + [0] * (max_length - len(prompt_token_ids))
            )
        return {
            "input_ids": torch.tensor(input_ids).to(device),
            "attention_mask": torch.tensor(attention_mask).to(device),
        }

    def encode(
        self,
        prompt_token_ids=None,
        num_logits_to_keep=0,
    ):        
        total_probs = []
        pbar = tqdm(total = len(prompt_token_ids), desc="Encoding")
        for index in range(len(prompt_token_ids)):
            batch_prompt_token_ids = prompt_token_ids[index:index+1]
            padded_batch = self._collate(batch_prompt_token_ids, device=self.model.device)
            with torch.no_grad():
                all_logits = self.model(
                    **padded_batch, 
                    return_dict=True, 
                    output_hidden_states=True, 
                    num_logits_to_keep=num_logits_to_keep,
                    use_cache=False,
                ) # a list of logits, each element = [num_layers, seq_len, vocab_size]
                num_layers = self.model.config.num_hidden_layers
                
                assert len(all_logits) == num_layers

                layer_probs = []
                for layer_index in range(num_layers):
                    local_logits = all_logits[layer_index] # [batch, seq_len, vocab_size]                    
                    shift_logits = F.softmax(local_logits[..., :-1, :], dim=2) #[batch, seq_len-1, vocab_size]

                    shift_labels = padded_batch["input_ids"][..., -shift_logits.shape[1]:].unsqueeze(2) #[batch, seq_len-1, 1]
                    probs = torch.gather(shift_logits, dim=2, index=shift_labels).squeeze(2).tolist() #[batch, seq_len-1]

                    assert len(probs) == len(batch_prompt_token_ids)
                    real_seq_lens = padded_batch["attention_mask"].sum(dim=1).tolist()
                    for local_index in range(len(probs)):
                        probs[local_index] = [0.0 for _ in range(padded_batch["attention_mask"].shape[1] - 1 - len(probs[local_index]))] + probs[local_index]
                        probs[local_index] = probs[local_index][:real_seq_lens[local_index]-1]

                    layer_probs.append(probs) # layer_probs = [layer_0_probs, layer_1_probs, ...], where layer_n_probs = [batch, seq_len-1]

            batch_probs = [] # should be [index_1_probs, index_2_probs, ...], where index_n_probs = [num_layer, seq_len-1]
            for sub_index in range(len(batch_prompt_token_ids)):
                index_n_probs = []
                for layer_index in range(num_layers):
                    index_n_probs.append(layer_probs[layer_index][sub_index])
                batch_probs.append(index_n_probs)
            
            assert len(batch_probs) == len(batch_prompt_token_ids)
            assert len(batch_probs[0]) == num_layers

            total_probs += batch_probs
            pbar.update(len(batch_probs))

        pbar.close()
        assert len(total_probs) == len(prompt_token_ids)

        total_results = []
        for one_prompt_token_ids, one_prompt_prob in zip(prompt_token_ids, total_probs):
            one_prompt_prob = [layer_probs[:len(one_prompt_token_ids)-1] for layer_probs in one_prompt_prob]
            results = {
                "prompt_token_ids": one_prompt_token_ids, # [seq_len]
                "multi_layer_prompt_probs": one_prompt_prob, # [num_layers, seq_len-1]
                "outputs": None
            }
            total_results.append(results)
            
        return total_results