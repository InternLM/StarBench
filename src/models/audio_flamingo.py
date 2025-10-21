import torch
import llava
from llava import conversation as clib
from llava.media import Image, Video, Sound
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat
from peft import PeftModel
from .base import BaseModel


class AudioFlamingo3(BaseModel):
    NAME = 'audio-flamingo-3'
    def __init__(self, model_path='nvidia/audio-flamingo-3', think=False, **kwargs):
        assert model_path is not None
        model_think = os.path.join(model_path, 'stage35')
        conv_mode = 'auto'
        model = llava.load(model_path)
        if think:
            model = PeftModel.from_pretrained(
                model,
                model_think,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        self.model = model.to("cuda")
        self.think = think
        torch.cuda.empty_cache() 


    def generate_inner(self, msgs):
        meta = msgs.get('meta', None)
        prompts = msgs.get('prompts', None)
        content = []
        #https://github.com/NVIDIA/audio-flamingo/blob/audio_flamingo_3/llava/cli/infer_audio.py#L79
        for x in prompts:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'audio':
                content.append(Sound(x['value']))
        if self.think:
            content.append(' Please think and reason about the input audio before you respond.')

        print(f'messages: {content}')
        output = self.model.generate_content(content, response_format=None)
        print('【af3 output】:', output)
        return output