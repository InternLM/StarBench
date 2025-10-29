import torch
import sys
from .base import BaseModel


class StepAudio2(BaseModel):
    NAME = 'step-audio2'
    def __init__(self, model_path='stepfun-ai/Step-Audio-2-mini', model_folder='./Step-Audio2', **kwargs):
        assert (model_path is not None) and (model_folder is not None)
        abs_model_folder= os.path.abspath(model_folder)
        sys.path.append(abs_model_folder)
        from stepaudio2 import StepAudio2
        # from token2wav import Token2wav
        self.model = StepAudio2(model_path)
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
        
        #https://github.com/stepfun-ai/Step-Audio2/blob/main/examples.py#L148
        messages = [
            {"role": "system", "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."},
            {'role': 'user', 'content': content},
            {"role": "assistant", "content": None}
        ]

        print(f'messages: {messages}')

        max_new_tokens = 256
        if meta and 'reasoning' in meta['task'].lower():
            max_new_tokens = 1024
        with torch.no_grad():
            tokens, output, _ = self.model(messages, max_new_tokens=max_new_tokens, num_beams=2)
        print('【step-audio-2 output】:', output)

        return output