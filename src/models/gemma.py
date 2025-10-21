import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from .base import BaseModel


class Gemma3n(BaseModel):
    NAME = 'gemma-3n-E4B-it'
    def __init__(self, model_path='google/gemma-3n-E4B-it', **kwargs):
        assert model_path is not None
        self.model =Gemma3nForConditionalGeneration.from_pretrained(model_path , device_map="cuda", torch_dtype=torch.bfloat16,).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
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
                    "audio": x['value'],
                })
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {'role': 'user', 'content': content}
        ]

        print(f'messages: {messages}')

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        input_len = inputs["input_ids"].shape[-1]

        max_new_tokens = 256
        if meta and 'holistic' in meta['task']: #NOTE
            max_new_tokens = 1024
        
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generation = generation[0][input_len:]
            output = self.processor.decode(generation, skip_special_tokens=True)
            print('【gemma output】:', output)    
        return output