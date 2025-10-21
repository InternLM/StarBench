import random
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from .base import BaseModel


class MiDashengLM(BaseModel):
    NAME = 'midashenglm-7b'
    def __init__(self, model_path='mispeech/midashenglm-7b', **kwargs):
        assert model_path is not None
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        torch.cuda.empty_cache() 

    def generate_inner(self, msgs):
        meta = msgs.get('meta', None)
        prompts = msgs.get('prompts', None)
        content = []
        for x in prompts:
            if x['type'] == 'text':
                content.append({"type": "text", "text": x['value']})
            elif x['type'] == 'audio':
                content.append({
                "type": "audio",
                "path": x['value'],
            })
        # https://github.com/xiaomi-research/dasheng-lm
        messages = [
                {
                    "role": "system",
                    "content": [ #https://github.com/xiaomi-research/dasheng-lm/issues/25 ??
                        {"type": "text", "text":"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",} # "You are a helpful language and speech assistant."
                    ],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]

        print_once(f'messages: {messages}')

        model_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                add_special_tokens=True,
                return_dict=True,
            )
        model_inputs = model_inputs.to(self.model.device).to(self.model.dtype)
        
        # max_new_tokens = 256
        # if meta and 'holistic' in meta['task']: 
        #     max_new_tokens = 1024

        with torch.no_grad():
            generation = self.model.generate(**model_inputs)
            output = self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0] 
            print('【midasheng output】:', output)    
        return output