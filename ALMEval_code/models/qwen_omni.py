import torch
from .base import BaseModel

class Qwen2_5Omni(BaseModel):
    NAME = 'qwen25-omni'
    def __init__(self, model_path='Qwen/Qwen2.5-Omni-7B', **kwargs):
        assert model_path is not None
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        except ImportError as e:
            raise ImportError(
                "❌ Failed to import Qwen2.5 Omni dependencies.\n"
                "Please make sure you have installed the correct transformers version"
            ) from e

        self.model_path = model_path
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2").eval()
        self.model.disable_talker()
        torch.cuda.empty_cache() 


    def generate_inner(self, msgs):
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError as e:
            raise ImportError(
                    "❌ Failed to 'from qwen_omni_utils import process_mm_info'"
                ) from e
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

        # print(f'messages: {messages}')

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
        if meta and 'reasoning' in meta['task'].lower():
            max_new_tokens = 1024

        with torch.no_grad():
            model_output = self.model.generate(
                **inputs,
                use_audio_in_video=False,
                thinker_max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                return_audio=False,
                thinker_do_sample=False,
                # repetition_penalty=1.0
            )
            output = self.processor.batch_decode(
                    model_output.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            # print('【omni output】:', output)    
        return output


class Qwen3Omni(BaseModel):
    NAME = 'qwen3-omni'
    def __init__(self, model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct", **kwargs):
        try:
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        except ImportError as e:
            raise ImportError(
                "❌ Failed to import Qwen2.5 Omni dependencies.\n"
                "Please make sure you have installed the correct transformers version"
            ) from e
        assert model_path is not None
        self.model_path = model_path
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2").eval()
        self.model.disable_talker()
        torch.cuda.empty_cache() 
    
    def generate_inner(self, msgs):
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError as e:
            raise ImportError(
                    "❌ Failed to 'from qwen_omni_utils import process_mm_info'"
                ) from e
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

        # print(f'messages: {messages}')

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
        if meta and 'reasoning' in meta['task'].lower():
            max_new_tokens = 1024

        with torch.no_grad():
            text_ids, audio = self.model.generate(
                **inputs,
                use_audio_in_video=False,
                # thinker_max_new_tokens=max_new_tokens,
                thinker_return_dict_in_generate=True, # for qwen3-omni
                # return_audio=False,
                # thinker_do_sample=False,
                # repetition_penalty=1.0
            )
            output = self.processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            # print('【omni output】:', output)    
        return output
