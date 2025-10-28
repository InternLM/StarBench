import sys
import torch
from .base import BaseModel
import os


class DeSTA25_Audio(BaseModel):
    NAME = 'desta'
    def __init__(self, model_path='DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B', model_folder='./DeSTA2.5-Audio', **kwargs):
        assert (model_path is not None) and (model_folder is not None)
        abs_model_folder= os.path.abspath(model_folder)
        sys.path.append(abs_model_folder)
        from desta import DeSTA25AudioModel
        self.model = model = DeSTA25AudioModel.from_pretrained(model_path).to("cuda")
        torch.cuda.empty_cache() 


    def generate_inner(self, msgs):
        meta = msgs.get('meta', None)
        prompts = msgs.get('prompts', None)
        text_query = ""
        audios = []
        for x in prompts:
            if x['type'] == 'text':
                text_query += x['value']
            elif x['type'] == 'audio':
                text_query += "<|AUDIO|>"
                audios.append({
                    "audio": x['value'],
                    "text": None,
                })
        messages = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions."
            },
            {
                "role": "user",
                "content": text_query,
                "audios": audios
            }
        ]

        print(f'messages: {messages}')

        max_new_tokens = 256
        if meta and 'reasoning' in meta['task'].lower():
            max_new_tokens = 1024

        with torch.no_grad():
            outputs = self.model.generate(
                messages=messages,
                do_sample=False,
                top_p=1.0,
                temperature=1.0,
                max_new_tokens=max_new_tokens
            )
            output = outputs.text[0]
            print('【desta25 output】:', output)

        return output