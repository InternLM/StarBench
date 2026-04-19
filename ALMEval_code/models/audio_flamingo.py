import os
import tempfile
import torch
from .base import BaseModel
from datasets.starbench import merge_pydub

# Audio Flamingo 3 HF version, you'd better update transformers >= 5.0
class AudioFlamingo3HF(BaseModel):
    NAME = 'audio-flamingo-3'
    def __init__(self, model_path='nvidia/audio-flamingo-3-hf', thinking=False, **kwargs):
        assert model_path is not None
        try:
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "❌ Failed to import AudioFlamingo3 dependencies.\n"
                "Please make sure you have installed the correct transformers version"
            ) from e

        self.model_path = model_path
        self.thinking = thinking

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        ).eval()

        # thinking: non_lora_trainables + LoRA（PEFT）
        if self.thinking:
            from peft import PeftModel

            # 1) Local Dir
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "think", "non_lora_trainables.bin")):
                snapshot_dir = model_path
                non_lora_path = os.path.join(model_path, "think", "non_lora_trainables.bin")
            else:
                # 2) get true path
                from cached_path import cached_path
                non_lora_path = str(cached_path(f"hf://{model_path}/think/non_lora_trainables.bin"))
                # non_lora_path: .../snapshots/<hash>/think/non_lora_trainables.bin
                snapshot_dir = os.path.dirname(os.path.dirname(non_lora_path))  # -> .../snapshots/<hash>

            # 3) load non_lora_trainables
            non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
            self.model.load_state_dict(non_lora_trainables, strict=False)

            # 4) load LoRA（从 snapshot_dir/think）
            self.model = PeftModel.from_pretrained(self.model, snapshot_dir, subfolder="think").eval()

        torch.cuda.empty_cache()


    def generate_inner(self, msgs):
        meta = msgs.get('meta', None)
        prompts = msgs.get('prompts', None)

        # Collect text and audio items separately
        text_parts = []
        audio_paths = []
        for x in prompts:
            if x['type'] == 'text':
                text_parts.append(x['value'])
            elif x['type'] == 'audio':
                audio_paths.append(x['value'])

        # HF processor requires text:audio = 1:1
        # If multiple audios, merge them into one with 2s silence gaps
        merged_path = None
        if len(audio_paths) > 1:
            merged_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            merge_pydub(audio_paths, merged_path)
            audio_paths = [merged_path]

        # Build content: all text first, then single audio
        combined_text = "\n".join(text_parts)
        if self.thinking:
            combined_text += "\nPlease think and reason about the input audio before you respond."

        content = [
            {"type": "text", "text": combined_text},
            {"type": "audio", "path": audio_paths[0]},
        ]
        messages = [{'role': 'user', 'content': content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = inputs.to(self.model.device)
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(self.model.dtype)

        max_new_tokens = 500
        if self.thinking or (meta and 'reasoning' in meta['task'].lower()):
            max_new_tokens = 1024

        with torch.no_grad():
            model_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
            )
            output = self.processor.batch_decode(
                    model_output.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
        if merged_path and os.path.exists(merged_path):
            os.remove(merged_path)

        print('【audio flamingo3 output】:', output)
        return output
