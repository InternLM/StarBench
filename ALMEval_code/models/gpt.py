import sys
import torch
from .base import BaseModel
import os
import base64

class GPT4o_Audio(BaseModel):
    NAME = 'gpt4o_audio'
    def __init__(self, model_id='gpt-4o-audio-preview', base_url=None, api_key=None, retry= 2, **kwargs):
        assert (model_id is not None) and (base_url is not None) and (api_key is not None)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                    "❌ Failed to 'from openai import OpenAI'"
                ) from e
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_id = model_id
        self.retry = retry

    def generate_inner(self, msgs, audio_type='wav'):
        meta = msgs.get('meta', None)
        prompts = msgs.get('prompts', None)
        text_query = ""
        audios = []
        content = []
        for x in prompts:
            if x['type'] == 'text':
                content.append({"type": "text", "text": x['value']})
            elif x['type'] == 'audio':
                with open(x['value'], 'rb') as f:
                    audio_data= base64.b64encode(f.read()).decode('utf-8')
                content.append({
                    "type": "input_audio", # NOTE: update this field to match your actual data format
                    "input_audio": {
                        "data": audio_data,
                        "format": audio_type
                        }
                })
        messages = [{'role': 'user', 'content': content}]

        # print(f'messages: {messages}')

        # --- Retry logic ---
        for attempt in range(1, self.retry + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages
                )
                output = response.choices[0].message.content
                print('【gemini output】:', output)
                return output
            except Exception as e:
                print(f"[Gemini] ⚠️ Attempt {attempt}/{self.retry} failed: {e}")
                if attempt < self.retry:
                    time.sleep(2 * attempt)  # exponential backoff
                else:
                    raise RuntimeError(f"[Gemini] ❌ All {self.retry} attempts failed.")

        return output