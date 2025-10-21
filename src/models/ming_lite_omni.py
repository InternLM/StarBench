#TO CHECK
import torch
from transformers import AutoProcessor, GenerationConfig
from .base import BaseModel
import warnings
warnings.filterwarnings("ignore")


class MingLiteOmni(BaseModel):
    NAME = 'Ming-Lite-Omni-1.5'
    def __init__(self, model_folder='./Ming', **kwargs):
        abs_model_folder= os.path.abspath(model_folder)
        sys.path.insert(0, abs_model_folder)
        model_path = os.path.join(model_folder, 'inclusionAI/Ming-Lite-Omni-1.5')
        from modeling_bailingmm import BailingMMNativeForConditionalGeneration
        
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
            attn_implementation="flash_attention_2",
            load_image_gen=False,
            low_cpu_mem_usage=True ,      # Minimize CPU memory during loading
            local_files_only=True
        ).to("cuda").eval()
        self.model.talker.use_vllm = False
        self.processor = AutoProcessor.from_pretrained(abs_model_folder, trust_remote_code=True, local_files_only=True) 
        torch.cuda.empty_cache() 


    def generate_inner(self, msgs):
        from test_audio_tasks import generate 
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
        messages = [{"role": "HUMAN", 'content': content}]
        print(f'messages: {messages}')
        output = generate(messages=messages, processor=self.processor, model=self.model)
        print('【ming omni output】:', output)
        return output