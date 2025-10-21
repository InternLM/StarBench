import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from .base import BaseModel
import librosa


class Qwen2AudioInstruct(BaseModel):
    NAME = 'Qwen2-Audio-7B-Instruct'
    def __init__(self, model_path='Qwen/Qwen2-Audio-7B-Instruct', **kwargs):
        assert model_path is not None
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map='cuda').eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        torch.cuda.empty_cache() 


    def generate_inner(self, msgs):
        meta = msgs.get('meta', None)
        prompts = msgs.get('prompts', None)
        content = []
        audio_data = []
        for x in prompts:
            if x['type'] == 'text':
                content.append({"type": "text", "text": x['value']})
            elif x['type'] == 'audio':
                content.append({
                    "type": "audio",
                    "audio_url": x['value'],
                })
                audio_data.append(librosa.load(x['value'], sr=self.processor.feature_extractor.sampling_rate)[0])
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, #Audio Analysis Inference
                    {'role': 'user', 'content': content}]

        print(f'messages: {messages}')

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=text,
            audio=audio_data, #https://github.com/QwenLM/Qwen2-Audio/issues/146
            return_tensors='pt',
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        max_new_tokens = 256
        if meta and 'holistic' in meta['task']: #NOTE
            max_new_tokens = 1024

        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            output = self.processor.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]  
            print('【qwen2-audio output】:', output)
        return output