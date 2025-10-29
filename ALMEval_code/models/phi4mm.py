import torch
from .base import BaseModel


class Phi4Multimodal(BaseModel):
    NAME = 'Phi4mm'
    def __init__(self, model_path='microsoft/Phi-4-multimodal-instruct', **kwargs):
        assert model_path is not None
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        except ImportError as e:
            raise ImportError(
                "❌ Failed to import Phi-4-multimodal-instruct dependencies.\n"
                "Please make sure you have installed the correct transformers version"
            ) from e
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True, # Allow execution of custom code from the model repository
            local_files_only=True,  # Force loading from local files only; remove if online access is available
            torch_dtype='auto',
            _attn_implementation='flash_attention_2',
        ).cuda()
        self.generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')
        torch.cuda.empty_cache() 


    def generate_inner(self, msgs):
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'

        meta = msgs.get('meta', None)
        prompts = msgs.get('prompts', None)
        text_query = user_prompt
        audio_idx = 0
        audios = []
        for x in prompts:
            if x['type'] == 'text':
                text_query += x['value']
            elif x['type'] == 'audio':
                audio_idx += 1
                text_query += f"<|audio_{audio_idx}|>"
                audios.append(soundfile.read(x['value']))
        text_query += f'\n{prompt_suffix}{assistant_prompt}' 
        
        print(f'text query: {text_query}')
        max_new_tokens = 256
        if meta and 'reasoning' in meta['task'].lower():
            max_new_tokens = 1024

        inputs = processor(text=prompt, audios=audios, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                generation_config=self.generation_config,
            )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
        output = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print('【phi4-mm output】:', output)  
        return output