import torch
from .base import BaseModel
import os
import sys

class KimiAudioInstruct(BaseModel):
    NAME = 'kimi'
    def __init__(self, model_path='moonshotai/Kimi-Audio-7B-Instruct', model_folder='./Kimi-Audio', **kwargs):
        assert (model_path is not None) and (model_folder is not None)
        abs_model_folder= os.path.abspath(model_folder)
        sys.path.append(abs_model_folder)
        try:
            from kimia_infer.api.kimia import KimiAudio
        except ImportError as e:
            raise ImportError(
                "❌ Failed to import kimi dependencies.\n"
                "Please 'pip install git+https://github.com/MoonshotAI/Kimi-Audio.git'"
            ) from e
        # from token2wav import Token2wav
        self.model = KimiAudio(
            model_path=model_path,
            load_detokenizer=False, #True,
        )
        torch.cuda.empty_cache() 


    def generate_inner(self, msgs):
        meta = msgs.get('meta', None)
        max_new_tokens = 256
        if meta and 'reasoning' in meta['task'].lower():
            max_new_tokens = 1024

        sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
            'max_new_tokens': 256,
        }

        prompts = msgs.get('prompts', None)
        messages = []
        for x in prompts:
            if x['type'] == 'text':
                messages.append({"role": "user", "message_type": "text", "content": x['value']})
            elif x['type'] == 'audio':
                messages.append({
                    "role": "user",
                    "message_type": "audio",
                    "content": x['value'],
                })
        print(f'messages: {messages}')
        with torch.no_grad():
            wav, output = self.model.generate(messages, **sampling_params, output_type="text")
        print('【kimi output】:', output)
        return output