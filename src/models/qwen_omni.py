import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from .base import BaseModel


class Qwen2_5Omni(BaseModel):
    NAME = 'Qwen2.5-Omni-7B'
    def __init__(self, model_path='Qwen/Qwen2.5-Omni-7B', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2").eval()
        self.model.disable_talker()
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
        messages = [{'role': 'user', 'content': content}]

        print(f'messages: {messages}')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        inputs = self.processor(text=text,
                                audio=audios,
                                images=images,
                                videos=videos,
                                return_tensors='pt',
                                padding=True, 
                                use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        max_new_tokens = 256
        if meta and 'holistic' in meta['task']: #NOTE
            max_new_tokens = 1024

        with torch.no_grad():
            model_output = self.model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                thinker_max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                return_audio=False,
                thinker_do_sample=False,
                # repetition_penalty=1.0
            )
            output = processor.batch_decode(
                    model_output.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            print('【omni output】:', output)    
        return output