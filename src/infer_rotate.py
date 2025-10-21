# 评测 AV-Odyssey 的 DeafTest
# 参考代码：https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/evaluate_chat.py

import json
import os
import torch
from tqdm import tqdm
import numpy as np
import argparse
import re
import glob
import sys
import base64
import librosa
import soundfile

os.environ['HF_HOME'] = '/mnt/shared-storage-user/liuzihan/.cache/huggingface'

# ========== 选项旋转工具 ==========
LETTER_ORDER = ['<A>', '<B>', '<C>', '<D>']

def rotate_letter(letter: str, r: int) -> str:
    """顺时针旋转 r 位：展示用（canonical -> rotated）。"""
    if letter not in LETTER_ORDER:
        return letter
    i = LETTER_ORDER.index(letter)
    return LETTER_ORDER[(i + (r % 4)) % 4]

def inverse_rotate_letter(letter: str, r: int) -> str:
    """逆旋转 r 位：把“展示字母/模型输出字母”映回 canonical。"""
    if letter not in LETTER_ORDER:
        return letter
    i = LETTER_ORDER.index(letter)
    return LETTER_ORDER[(i - (r % 4)) % 4]

def rotate_options_map(options_map: dict, r: int) -> dict:
    """
    给定 canonical 的 {<A>: textA, <B>: textB, ...}，
    生成旋转后的“展示”映射：仍用 <A>/<B>/<C>/<D> 作为外层 key，
    但每个展示位上的文本来自 canonical 中对应逆旋转的来源。
    例如 r=1 时：展示位 <A> 放 canonical 的 <D> 文本，<B> 放 canonical 的 <A> 文本，依此类推。
    """
    rotated = {}
    for disp_letter in LETTER_ORDER:
        canonical_src = inverse_rotate_letter(disp_letter, r)  # 展示位对应的 canonical 源
        rotated[disp_letter] = options_map[canonical_src]
    return rotated

# ========== 其他工具 ==========
def enable_proxy():  # 需修改
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''

#NOTE 替换infer代码中的原parse_multi_choice_response, 注意删除下面return中的第二个值（只是方便检验）
def parse_multi_choice_response(response: str, all_choices: list, index2ans: dict) -> str:
    """
    根据一个优化的优先级策略，从模型生成的响应中解析出多项选择题的预测答案。

    优先级顺序:
    1. 查找由关键词引导的最后一个选项字母。
    2. 如果没有，则依次匹配以下选项:
        优先级2: 有特殊符号包括的选项lette, (A), <A>, [A]  NOTE 故此处强烈建议用<A>这样的格式包裹选项
        优先级1: 匹配选项文本 
        优先级0: 无特殊符号包裹的letter, A,B,C (易与文本中的其他单词混淆，故优先级最低)
        若有匹配到的，则按（优先级，出现位置排列）， 即优先级最高，且出现在最后的选项
        eg:[('<A>', 220, 0), ('<C>', 28, 1), ('<A>', 435, 1), ('<C>', 663, 1), ('<C>', 24, 2)], 最后的输出为('<C>', 24, 2    
    3. 如果全部失败，返回默认值 'Z'。
    参数:
        response (str): 模型生成的文本响应。
        all_choices (list): 所有可能的选项字母列表，例如: ['A', 'B', 'C', 'D']。
        index2ans (dict): 一个将选项字母映射到其完整答案文本的字典。

    返回:
        str: 解析出的选项字母 (例如 '<A>')，如果无法解析则返回 'Z'。
    """
    
    choices_str = "".join(all_choices)

    # --- 优先级 1: 查找由关键词引导的最后一个选项字母 ---
    keyword_matches = []
    # 定义关键词和选项格式的组合模式
    # (?i) 表示不区分大小写, [\s:：]* 表示可选的空格或中英文冒号
    # 使用 \bA\b 来确保匹配的是独立的字母，而不是单词的一部分
    keyword_pattern = re.compile(
        r"(?i)(?:answer|<answer>|solution|choice|option|correct option is|answer is|答案是|选项是)\s*[:：]*\s*\*{{0,2}}"
        r"(?:"
        # 格式1: (A), <A>, [A], {A} #NOTE 新增支持 $\\boxed{F}$， gemini-2.5-flash喜欢这种格式
        # r"[\(\<\[]\s*([{choices}])\s*[\)\]\>]" 原来的 增加了{}
        r"[\(\<\[\{{]\s*([{choices}])\s*[\)\]\>\}}]"
        # 格式2: 独立的字母 A
        r"|\b([{choices}])\b"
        r")".format(choices=choices_str)
    )
    

    for match in re.finditer(keyword_pattern, response):
        # 选项字母可能在第一个或第二个捕获组中
        found_choice_first = match.group(1) 
        found_choice_second = match.group(2)
        if found_choice_second:
            keyword_matches.append((found_choice_second, match.start(), 0))  # (A), <A>, [A] 的优先级比 独立的 A  高！
        if found_choice_first:
            keyword_matches.append((found_choice_first, match.start(), 1))
            
    if keyword_matches:
        # 按出现位置排序，返回最后一个匹配的选项
        keyword_matches.sort(key=lambda x: (x[2], x[1]))
        # print('keyword_matches:', keyword_matches)
        return '<'+keyword_matches[-1][0]+'>'

    # 优先级: 2 :(A), <A>, [A] ; 1: 选项文本匹配;  0: 独立的 A 
    standalone_matches = []
    # 定义一个只包含选项格式的模式
    standalone_pattern = re.compile(
        r"[\(\<\[\{{]\s*([{choices}])\s*[\)\]\>\}}]" # 格式1: (A), <A>, [A], {A} #NOTE 新增支持 $\\boxed{F}$， gemini-2.5-flash喜欢这种格式
        r"|\b([{choices}])\b".format(choices=choices_str) # 独立的 A
    )

    for match in re.finditer(standalone_pattern, response):
        found_choice_first = match.group(1)  
        found_choice_second = match.group(2)
        if found_choice_first:
            standalone_matches.append(('<'+found_choice_first+'>', match.start(), 2))
        if found_choice_second:
            standalone_matches.append(('<'+found_choice_second+'>', match.start(), 0))  
        
    if len(response.split()) > 2:
        for choice_letter, choice_text in index2ans.items():
            # 使用 re.escape 来安全地匹配可能包含特殊字符的文本，并忽略大小写
            for match in re.finditer(re.escape(choice_text), response, re.IGNORECASE):
                standalone_matches.append((choice_letter, match.start(), 1))
    
    if standalone_matches:
        # 按出现位置排序，返回最后一个匹配的选项
        standalone_matches.sort(key=lambda x: (x[2], x[1])) # 先按优先级 (0 先于 1)，再按出现位置
        # print('standalone_matches:', standalone_matches)
        return standalone_matches[-1][0]

    return 'Z'


# def parse_multi_choice_response(response: str, all_choices: list, index2ans: dict) -> str:
#     """
#     根据一个优化的优先级策略，从模型生成的响应中解析出多项选择题的预测答案。

#     优先级顺序:
#     1. 查找由关键词引导的最后一个选项字母。
#     2. 如果没有，则查找文本中最后一个独立出现的选项标识。
#     3. 如果没有，则匹配答案的完整文本内容。
#         3.1. 如果只匹配到一个选项，直接返回该选项。
#         3.2. 【关键优化】如果匹配到多个选项，则启动“比较句解析”作为仲裁者来解决歧义
#     4. 如果全部失败，返回默认值 'Z'。

#     参数:
#         response (str): 模型生成的文本响应。
#         all_choices (list): 所有可能的选项字母列表，例如: ['A', 'B', 'C', 'D']。
#         index2ans (dict): 一个将选项字母映射到其完整答案文本的字典。

#     返回:
#         str: 解析出的选项字母 (例如 'A')，如果无法解析则返回 'Z'。
#     """
    
#     choices_str = "".join(all_choices)

#     # --- 优先级 1: 查找由关键词引导的最后一个选项字母 ---
#     keyword_matches = []
#     # 定义关键词和选项格式的组合模式
#     # (?i) 表示不区分大小写, [\s:：]* 表示可选的空格或中英文冒号
#     # 使用 \bA\b 来确保匹配的是独立的字母，而不是单词的一部分
#     keyword_pattern = re.compile(
#         r"(?i)(?:answer|<answer>|solution|choice|option|correct option is|answer is|答案是|选项是)\s*[:：]*\s*"
#         r"(?:"
#         # 格式1: (A), <A>, [A]
#         r"[\(\<\[]\s*([{choices}])\s*[\)\]\>]"
#         # 格式2: 独立的字母 A
#         r"|\b([{choices}])\b"
#         r")".format(choices=choices_str)
#     )

#     for match in re.finditer(keyword_pattern, response):
#         # 选项字母可能在第一个或第二个捕获组中
#         found_choice = match.group(1) or match.group(2)
#         if found_choice:
#             keyword_matches.append((found_choice, match.start()))
            
#     if keyword_matches:
#         # 按出现位置排序，返回最后一个匹配的选项
#         keyword_matches.sort(key=lambda x: x[1])
#         return '<'+keyword_matches[-1][0]+'>'

#     # --- 优先级 2: 查找文本中最后一个独立出现的选项标识 ---
#     standalone_matches = []
#     # 定义一个只包含选项格式的模式
#     standalone_pattern = re.compile(
#         r"[\(\<\[]\s*([{choices}])\s*[\)\]\>]"  # (A), <A>, [A]
#         r"|\b([{choices}])\b".format(choices=choices_str) # 独立的 A
#     )

#     for match in re.finditer(standalone_pattern, response):
#         found_choice = match.group(1) or match.group(2)
#         if found_choice:
#             standalone_matches.append((found_choice, match.start()))

#     if standalone_matches:
#         # 按出现位置排序，返回最后一个匹配的选项
#         standalone_matches.sort(key=lambda x: x[1])
#         return '<'+standalone_matches[-1][0]+'>'

#     # --- 优先级 3: 匹配答案的完整文本内容 ---
#     text_matches = []
#     # 避免在过短的响应上进行匹配，可能导致误判
#     if len(response.split()) > 2:
#         for choice_letter, choice_text in index2ans.items():
#             # 使用 re.escape 来安全地匹配可能包含特殊字符的文本，并忽略大小写
#             for match in re.finditer(re.escape(choice_text), response, re.IGNORECASE):
#                 text_matches.append((choice_letter, match.start()))
                
#     if text_matches:
#         # 同样，按位置排序并返回最后一个匹配项
#         text_matches.sort(key=lambda x: x[1])
#         return text_matches[-1][0]

#     # --- 优先级 4: 默认返回值 ---
#     return 'Z'


def qwen2_5Omni_process(audio_model, processor, data_inputs, answer_style, USE_AUDIO_IN_VIDEO=False):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style

        user_content = [
            {'type': 'audio', 'audio': audio_path},
            {'type': 'text', 'text': query},
            
        ]

        messages = [
            # {
            #     "role": "system",
            #     "content": [
            #         {"type": "text", "text": "You are a helpful assistant."}, #audio analysis 时默认是helpful assistant; 若需要speech output则为该prompt "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            #     ],
            # },
            {'role': 'user', 'content': user_content}]
        print('messages:', messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("text:", text)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(audio_model.device).to(audio_model.dtype)

        # github issue
        # model_output = audio_model.generate(
        #     **inputs,
        #     max_new_tokens=256,#120
        #     eos_token_id=processor.tokenizer.eos_token_id,
        #     do_sample=True,
        #     return_dict_in_generate=True,
        #     temperature=0.01,
        # )
        
        #参照kimi-audio-evalkit 吧 https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/master/almeval/models/qwen_omni.py#L146
        with torch.no_grad():
            model_output = audio_model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                thinker_max_new_tokens=256,
                return_dict_in_generate=True,
                return_audio=False,
                thinker_do_sample=False,
                # repetition_penalty=1.0
            )
            output = processor.batch_decode(
                model_output.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        print('omni output:', output)

        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data

def ming_omni_process(audio_model, processor, data_inputs, answer_style):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style

        #https://github.com/inclusionAI/Ming/tree/main#
        messages = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": query}
                ],
            },
        ]
        print('message:', messages)
        # 1. Format inputs using chat template
        output = ming_omni_generate(messages=messages, processor=processor, model=audio_model)
        print('ming omni output:', output)

        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data


def midashenglm_process(audio_model, processor, tokenizer, data_inputs, answer_style):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style

        # https://github.com/xiaomi-research/dasheng-lm
        messages = [
                {
                    "role": "system",
                    "content": [ #https://github.com/xiaomi-research/dasheng-lm/issues/25  ？？？
                        {"type": "text", "text":"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",} # "You are a helpful language and speech assistant."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        
                        {
                            "type": "audio",
                            "path": audio_path,
                            # or "url": "https://example.com/example.wav"
                            # or "audio": np.random.randn(16000)
                        },
                        {"type": "text", "text": query},
                    ],
                },
            ]
        print('messages:', messages)
        with torch.no_grad():
            model_inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                add_special_tokens=True,
                return_dict=True,
            )
            model_inputs = model_inputs.to(audio_model.device).to(audio_model.dtype)
            generation = audio_model.generate(**model_inputs)
            output = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]  # ["An engine is idling."]

        print('midasheng output:', output)

        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data

def desta25_process(audio_model, data_inputs, answer_style):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style

        # https://github.com/kehanlu/DeSTA2.5-Audio
        messages = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions."
            },
            {
                "role": "user",
                "content": "<|AUDIO|>\n"+query,
                "audios": [{
                    "audio": audio_path,  # Path to your audio file
                    "text": None
                }]
            }
        ]
        print('messages:', messages)
        with torch.no_grad():
            outputs = audio_model.generate(
                messages=messages,
                do_sample=False,
                top_p=1.0,
                temperature=1.0,
                max_new_tokens=512
            )
            output = outputs.text[0]

        print('desta25 output:', output)

        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data

def phi4mm_process(audio_model, generation_config, data_inputs, answer_style):
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style

        # https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_inference_phi4mm.py#L162
        prompt = f'{user_prompt}<|audio_1|>{query}{prompt_suffix}{assistant_prompt}'
        print('prompt:', prompt)
        audio = soundfile.read(audio_path)
        inputs = processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda:0')
        with torch.no_grad():
            generate_ids = audio_model.generate(
                **inputs,
                max_new_tokens=1000,
                generation_config=generation_config,
            )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
        output = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print('phi4-mm output:', output)

        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data


def gemma_process(audio_model, processor, data_inputs, answer_style):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style
        #https://huggingface.co/google/gemma-3n-E4B-it
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": query } #+ 'descripe the audio content first'
                ]
            }
        ]
        print('messages:', messages )

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(audio_model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = audio_model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generation = generation[0][input_len:]

        output = processor.decode(generation, skip_special_tokens=True)

        
        print('gemma output:', output)

        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data



def af3_process(audio_model, model_id, data_inputs, answer_style):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style

        if 'think' in model_id:
            query += ' Please think and reason about the input audio before you respond. Output must match exactly one of the listed choices.' 

        #https://github.com/NVIDIA/audio-flamingo/blob/audio_flamingo_3/llava/cli/infer_audio.py#L79
        prompts = [
            Sound(audio_path),
            query
        ]
        print('messages:', prompts)
        output = model.generate_content(prompts, response_format=None)
        print('af3 output:', output)
        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data


def salmonn_process(audio_model, processor, cfg,  model_id, data_inputs, answer_style):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style

        samples = prepare_one_sample(audio_path, processor)
        #https://github.com/bytedance/SALMONN/blob/salmonn/cli_inference.py#L52
        prompt = [
            cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + query.strip())
        ]
        print('messages:', prompt)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model.generate(samples, cfg.config.generate, prompts=prompt)[0]
       
        print('salmonn output:', output)
        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data



def qwen_audio_process(audio_model, processor, model_id, data_inputs, answer_style):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style

        audio_data = [librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)[0]]

        if 'instruct' in model_id:
            conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}, #Audio Analysis Inference
                            {'role': 'user',
                             'content': [{'type': 'audio', 'audio_url': audio_path},
                                         {'type': 'text', 'text': query}]}]
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )                             
        else:
            text = '<|audio_bos|><|AUDIO|><|audio_eos|>' + query
        
        print("text:", text)
        inputs = processor(
            text=text,
            audio=audio_data, #https://github.com/QwenLM/Qwen2-Audio/issues/146
            return_tensors='pt',
            padding=True,
            sampling_rate=processor.feature_extractor.sampling_rate
        )
        inputs = inputs.to(audio_model.device).to(audio_model.dtype)
        
        generate_ids = audio_model.generate(**inputs, max_new_tokens=256)#max_length=256
        # max_new_tokens=256, min_new_tokens=1, do_sample=False,
        #                                     top_k=None,
        #                                     top_p=None)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        output = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print('qwen2-audio output:', output)

        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data


def step_audio_process(audio_model, model_id, data_inputs, answer_style):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style


        if 'base' not in model_id:
            #https://github.com/stepfun-ai/Step-Audio2/blob/main/examples.py#L148
            messages = [
                {"role": "system", "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."},
                {"role": "human", "content": [{"type": "audio", "audio": audio_path},
                                            {"type": "text", "text": query}]},
                {"role": "assistant", "content": None}
            ]
        else:
            pass #text = '<|audio_bos|><|AUDIO|><|audio_eos|>' + query
        
        print("messages:", messages)
        tokens, output, _ = audio_model(messages, max_new_tokens=256, num_beams=2)

        print('step-audio-2 output:', output)

        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data

def kimi_audio_process(audio_model, data_inputs, answer_style):
    
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
    #构造query 和 audios
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        options = cur_data['options']
        option_text=""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style
        print(query)

        messages = [
            {
                "role": "user",
                "message_type": "audio",
                "content":audio_path
            },
            {"role": "user", "message_type": "text", "content": query},
            
        ]
        print('messages:', messages)
        wav, output = audio_model.generate(messages, **sampling_params, output_type="text")
        print('kimi output:', output)
        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data

def gemini_gpt_process(client, data_inputs, model_id, answer_style, audio_type='wav'): #"gemini-2.5-pro-preview-03-25" "gpt-4o-audio-preview-2024-12-17"  "gemini-2.0-flash"
    #构造query 和 audios
    result_data = []
    for cur_data in tqdm( data_inputs):
        audio_path = cur_data['audio']
        with open(audio_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
       
        options = cur_data['options']
        option_text = ""
        for i, Letter in enumerate(['A', 'B', 'C', 'D']):
            option_text += f"<{Letter}>: {options[i]}\n"
        question = cur_data['question']
        query = question + "\n" + option_text + answer_style
        
        user_content = [
            {
                'type': 'text',
                'text': query
            }
        ]

        print('query:', query)

        if 'gemini' in model_id: # or "gpt" in model_id:
            user_content.append({
                "type": "image_url",
                "image_url":f"data:audio/{audio_type};base64,{audio_data}"
            })
            messages = [
                {
                    "role": "user",
                    "content": user_content
                },
            ]
            response = client.chat.completions.create(
                model = model_id,
                messages = messages)
            output = response.choices[0].message.content
        else:
            
            user_content.append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": audio_type
                            }
                        })
            messages = [
                    {
                        "role": "user",
                        "content": user_content
                    },
                ]

            response = client.chat.completions.create(
                model=model_id,
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": audio_type},
                messages=messages
            )
            output = response.choices[0].message.audio.transcript
        print('Answer:', output)
        result_data.append({
                "question_id": cur_data['question_id'],
                "answer": cur_data['answer'],
                "model_output": output
            })
    return result_data


def load_comparison_data_from_disk(trials_dir: str, task_id: str, task_prompt: str, options_list: list):
    wav_files = glob.glob(os.path.join(trials_dir, f'*.wav'))
    if not wav_files: raise FileNotFoundError(f"No comparison files for '{task_id}' in '{trials_dir}'")

    structured_data = []
    # --- MODIFIED: 根据 task_id 选择正确的正则表达式 ---
    if task_id == "loudness_compare":
        pattern = re.compile(r"level(\d+)_(-?\d+)dBHL_(-?\d+)dBHL_(-?\d+)\.wav")
    elif task_id == "pitch_compare":
        pattern = re.compile(r"level(\d+)_(\d+)Hz_(\d+)Hz_(-?\d+)\.wav")
    elif task_id == "duration_compare":
        pattern = re.compile(r"level(\d+)_(\d+)ms_(\d+)ms_(-?\d+)\.wav")
    else:
        raise ValueError(f"Unknown task_id for comparison loader: {task_id}")

    for file_path in wav_files:
        data = {'audio': file_path, 'question': task_prompt, 'options': options_list}
        match = pattern.search(os.path.basename(file_path))
        
        if match:
            ans_list = ['<C>', '<A>', '<B>']
            level, v1, v2, ans_id = map(int, match.groups())
            # print('ans_id:', ans_id)
            data.update({
                    'task_id': task_id, 
                    'level':level,
                    'diff': abs(v1 - v2),
                    'value1': v1,
                    'value2': v2,
                    'answer': ans_list[ans_id] if level > 1 else '<C>',
                })
            data['question_id'] = os.path.basename(file_path).replace('.wav', '')
            structured_data.append(data)
            
    print(f"Loaded {len(structured_data)} {task_id} trials.")
    assert len(structured_data) ==480, f"len(structured_data): {len(structured_data)} !=480"
    return structured_data

def load_audiogram_data_from_disk(trials_dir: str, task_id: str, task_prompt: str, options_list: list):
    """
    (您的数据加载函数 - 现在只存储文件路径，不进行任何预处理)
    """
    wav_files = glob.glob(os.path.join(trials_dir, '*.wav'))
    if not wav_files:
        raise FileNotFoundError(f"No files for '{task_id}' in '{trials_dir}'")
        
    structured_data = []
    # pattern = re.compile(r"trial_(\d+)Hz_(-?\d+)dBHL_target-in-(\d)nd-interval\.wav")
    pattern = re.compile(r"(\d+)Hz_(-?\d+)dBHL_(\d)\.wav")
    
    for file_path in wav_files:
        match = pattern.search(os.path.basename(file_path))
        if match:
            pitch, db_level, correct_interval = map(int, match.groups())
            
            structured_data.append({
                # --- MODIFIED: 'audio' key now holds the direct file path ---
                'audio': file_path,
                'question': task_prompt,
                'options': options_list,
                'task_id': task_id,
                'answer': '<A>' if correct_interval == 1 else '<B>', #'<B>' if correct_interval == 1 else '<C>',  #
                'question_id': os.path.basename(file_path).replace('.wav', ''),
                'pitch': pitch,
                'db_level': db_level
            })
    return structured_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audiogram Test for Qwen-Omni')
    parser.add_argument('--model', default="gemini-2.5-pro", type=str, help='Model identifier') #gpt-4o-audio-preview-2025-06-03
    parser.add_argument('--tasks', type=list, default=['hearing_threshold','loudness_compare', 'pitch_compare', 'duration_compare'], help='A list of auditory tasks to evaluate in sequence.') # ,  'loudness_compare', 'pitch_compare', 'duration_compare' 'hearing_threshold','loudness_compare', 'pitch_compare', 'duration_compare'
    parser.add_argument('--base_data_dir', type=str, default='/mnt/shared-storage-user/liuzihan/ST_Benchmark/AttributeTest/data_samples', help='Directory with audiogram .wav files') #/fs-computility/mllm/liuzihan/Benchmark_eval/SensitivityTest/output
    parser.add_argument('--base_results_dir', type=str, default='/mnt/shared-storage-user/liuzihan/ST_Benchmark/AttributeTest/test_results', help='Directory with audiogram .wav files') 
    parser.add_argument('--num_seeds',  default=[40], help='Number of evaluation runs with different seeds')
    parser.add_argument('--opt_rotate', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='Rotate options by r in {0,1,2,3}. Support multiple values: e.g., --opt_rotate 0 1 2 3')
    parser.add_argument('--resume', action='store_true',
                    help='If set, skip tasks whose output jsonl already exists instead of overwriting.')
    args = parser.parse_args()
    model_name = args.model

    # --- 加载模型 (您的原始代码) ---
    if 'gemini' in model_name or 'gpt' in model_name:
        from openai import OpenAI
        base_url = "https://api.boyuerichdata.opensphereai.com/v1" #"http://35.220.164.252:3888/v1" # 'http://127.0.0.1:8005/v1/'  #
        api_key = "sk-TL4hrsr2r1XhIXJp0kibaqQFY73JViYAgErUZn4WapTZEaKm" 
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    # elif 'gpt' in model_name: #gpt-4o-audio-preview-2025-06-03 暂时不支持中转，只能直连
    #     from openai import OpenAI
    #     enable_proxy()# 开代理
    #     api_key = "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"
    #     client = OpenAI(api_key=api_key)
    elif 'qwen-omni' in model_name:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info
        #参考 https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb
        model_path = "/mnt/shared-storage-user/liuzihan/saved_ckpts/Qwen2.5-Omni-7B"  #"Qwen/Qwen2.5-Omni-7B" #
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2").eval() #"auto" #, attn_implementation="flash_attention_2"
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        model.disable_talker()
    elif "qwen2-audio" in model_name:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        if 'instruct' in model_name:
            model_path = "/mnt/shared-storage-user/liuzihan/saved_ckpts/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a" #"Qwen/Qwen2-Audio-7B-Instruct"
        else:
            model_path = "/mnt/shared-storage-user/liuzihan/saved_ckpts/models--Qwen--Qwen2-Audio-7B/snapshots/dd84470756e6277a71d4d7188773a43cde92696e" #'Qwen/Qwen2-Audio-7B'
        model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map='cuda').eval()
        processor = AutoProcessor.from_pretrained(model_path)

    elif 'kimi' in model_name:
        #虚拟环境换成 kimi, 需要flash-attn 和 gpu
        kimi_path = "/mnt/shared-storage-user/liuzihan/Kimi-Audio" #"/fs-computility/mllm/liuzihan/Kimi-Audio"
        #修改了KimiAPromptManager 中 Glm4Tokenizer的路径, 为本地路径 
        sys.path.append(kimi_path)
        from kimia_infer.api.kimia import KimiAudio
        import flash_attn
        model_path = "/mnt/shared-storage-user/liuzihan/saved_ckpts/huoshan_cache/huggingface/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b" #'/mnt/shared-storage-user/liuzihan/saved_ckpts/models--moonshotai--Kimi-Audio-7B/snapshots/d90be7113cd5be6bcf54fae2aabcad060a23f6cf'
        model = KimiAudio(
            model_path=model_path,
            load_detokenizer=False, #True,
        ).to('cuda').eval()
    elif 'step' in model_name:
        step_path= "/mnt/shared-storage-user/liuzihan/Step-Audio2"
        sys.path.append(step_path)
        from stepaudio2 import StepAudio2
        from token2wav import Token2wav
        model_path = '/mnt/shared-storage-user/liuzihan/saved_ckpts/models--stepfun-ai-Step-Audio-2-mini'
        model = StepAudio2(model_path)
    elif 'midasheng' in model_name:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        model_id = "/mnt/shared-storage-user/liuzihan/saved_ckpts/models--mispeech--midashenglm-7b"
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    elif 'desta25' in model_name: 
        #numpy >2.0 时需修改一下源代码 
        desta_path = '/mnt/shared-storage-user/liuzihan/DeSTA2.5-Audio'
        sys.path.append(desta_path)
        from desta import DeSTA25AudioModel
        #注意修改config文件和modeling文件里whisper和llm的路径
        model_path = '/mnt/shared-storage-user/liuzihan/saved_ckpts/models--DeSTA-ntu--DeSTA2.5-Audio-Llama-3.1-8B'
        model = DeSTA25AudioModel.from_pretrained(model_path).to("cuda")
    elif 'af3' in model_name:
        #虚拟环境需换到af3
        # af3_path = '/mnt/shared-storage-user/liuzihan/audio-flamingo'
        # sys.path.append(af3_path)
        # export TRITON_CACHE_DIR=/mnt/shared-storage-user/liuzihan/.triton
        import llava
        from llava import conversation as clib
        from llava.media import Image, Video, Sound
        from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat
        from peft import PeftModel

        model_path = '/mnt/shared-storage-user/liuzihan/saved_ckpts/models--nvidia--audio-flamingo-3'
        model_think = os.path.join(model_path, 'stage35')
        conv_mode = 'auto'
        model = llava.load(model_path)
        if 'think' in model_name:
            print('load think model!')
            model = PeftModel.from_pretrained(
                model,
                model_think,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        model = model.to("cuda")
        # clib.default_conversation = clib.conv_templates[conv_mode].copy()
    elif 'salmonn' in model_name:
        salmonn_path = '/mnt/shared-storage-user/liuzihan/SALMONN'
        sys.path.append(salmonn_path)
        # https://github.com/bytedance/SALMONN/blob/salmonn/cli_inference.py
        from transformers import WhisperFeatureExtractor
        from config import Config
        from models.salmonn import SALMONN
        from utils import prepare_one_sample
        import omegaconf

        cfg_path = "/mnt/shared-storage-user/liuzihan/SALMONN/configs/decode_config.yaml"
        device = "cuda"
        cfg = Config(Namespace(cfg_path=cfg_path, device=device, options=None)) 
        model = SALMONN.from_config(cfg.config.model).to(device).eval()
        processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    
    elif 'phi4-mm' in model_name:
        #peft 从 0.15.2 改成 0.13.2
        #transforms 版本> 0.50 会报错, 故用虚拟环境 af3
        # https://huggingface.co/microsoft/Phi-4-multimodal-instruct/commit/14643ff610c07a71fc8f984ca60ac8205d0560e4
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        model_path = '/mnt/shared-storage-user/liuzihan/saved_ckpts/models--microsoft--Phi-4-multimodal-instruct'
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True, # 允许用仓库里的自定义代码
            local_files_only=True,  # 强制只从本地加载，不联网； 若能联网可删除
            torch_dtype='auto',
            _attn_implementation='flash_attention_2',
        ).cuda()
        generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')
    
    elif 'gemma' in model_name:
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        model_path = '/mnt/shared-storage-user/liuzihan/saved_ckpts/models--google--gemma-3n-E4B-it/snapshots/c1221e9c62e34a43ab7ffacd1be0ea71f126ef10'
        model = Gemma3nForConditionalGeneration.from_pretrained(model_path , device_map="cuda", torch_dtype=torch.bfloat16,).eval()
        processor = AutoProcessor.from_pretrained(model_path)
    elif 'ming-omni' in model_name: #暂不work
        # https://github.com/inclusionAI/Ming/blob/main/cookbook.ipynb
        # diffusers==0.33.0 原环境是0.35.1.  transformers==4.52.4
        import warnings
        warnings.filterwarnings("ignore")
        ming_path = '/mnt/shared-storage-user/liuzihan/Ming'
        sys.path.insert(0, "/mnt/shared-storage-user/liuzihan/Ming")
        os.environ["HF_HOME"] = "/mnt/shared-storage-user/liuzihan/.cache/huggingface"
        from transformers import AutoProcessor, GenerationConfig
        from modeling_bailingmm import BailingMMNativeForConditionalGeneration
        from test_audio_tasks import generate as ming_omni_generate
        model_path = '/mnt/shared-storage-user/liuzihan/Ming/inclusionAI/Ming-Lite-Omni-1.5'
        model = BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
            attn_implementation="flash_attention_2",
            load_image_gen=False,
            low_cpu_mem_usage=True ,      # Minimize CPU memory during loading
            local_files_only=True
        ).to("cuda").to(torch.bfloat16)  # Run on GPU
        model.talker.use_vllm = False
        processor = AutoProcessor.from_pretrained(ming_path, trust_remote_code=True, local_files_only=True)
    
    
    elif 'baichuan-omni' in model_name: #代码不清楚
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model.training = False
        model.bind_processor(tokenizer, training=False, relative_path="/")












    # elif 'baichuan' in model_name:  "baichuan" "step_audio"
    #加了answer_style qwen_omni的结果会变差    
    answer_style = "" #"Answer with the option's letter from the given choices directly." #"Please state the option's letter of your answer clearly at the end of your response." 
    TASK_CONFIGS = {
        'hearing_threshold': {
            'trials_dir': os.path.join(args.base_data_dir, 'audiogram_test/audiogram_samples'), # 
            'prompt': "You are a helpful assistant in a hearing test. The audio you just heard is divided into two halves. Does a sound appear in the first half, the second half, or is it not present at all?",  ## sound The audio you just heard is divided into two halves. Does a beep appear in the first half, the second half, or is it not present at all? #the specific sound "You will take a hearing test. We will provide an audio clip that contains a specific **test tone** (with a known frequency and loudness). The test tone may appear in the **first half** of the audio, the **second half**, or there may be no test tone at all.\nPlease listen to the following audio and select the position where the test tone appears:", # Your prompt here
            'options_map': {'<A>': "The first half", '<B>': "The second half", '<C>': "It is not present at all", '<D>': "Unable to determine"},
            # 'options_map': {'<A>': "It is not present at all", '<B>': "The first half", '<C>': "The second half",  '<D>': "Unable to determine"},
            'answer_style': answer_style, 
            'loader_function': load_audiogram_data_from_disk #TODO 
        },
        'loudness_compare': {
            'trials_dir': os.path.join(args.base_data_dir, 'sensitivity_test/audios/loudness_compare/comparison_samples'), # 
            'prompt': "You are a helpful assistant in a loudness test. Which sound is louder: the first sound, the second sound, or are they the same?", #"Which audio has greater loudness? The first audio or the second audio?", #"You will listen to an audio clip containing two test tones. Which is louder?", You are a helpful assistant in a loudness test. 
            'options_map': { '<A>': "The first sound is louder", '<B>': "The second sound is louder", '<C>': "Both sounds are the same", '<D>': "Unable to determine"}, #{'<A>': "The first test tone.", '<B>': "The second test tone.", '<C>': "Unable to determine."}
            'answer_style': answer_style, 
            'loader_function': load_comparison_data_from_disk
        },
        'pitch_compare': {
            'trials_dir': os.path.join(args.base_data_dir, 'sensitivity_test/audios/pitch_compare/comparison_samples'), #
            'prompt': "You are a helpful assistant in a pitch test. Which sound has a higher pitch: the first sound, the second sound, or are they the same?", #"Which audio has a higher pitch? The first audio or the second audio?", #"You will listen to an audio clip containing two test tones. Which is higher in pitch?",
            'options_map': {'<A>': "The first sound has a higher pitch", '<B>': "The second sound has a higher pitch", '<C>': "Both sounds are the same",'<D>': "Unable to determine"},#{'<A>': "The first test tone.", '<B>': "The second test tone.", '<C>': "Unable to determine."}
            'answer_style': answer_style, 
            'loader_function': load_comparison_data_from_disk
        },
        'duration_compare': {
            'trials_dir': os.path.join(args.base_data_dir, 'sensitivity_test/audios/duration_compare/comparison_samples'), #
            'prompt': "You are a helpful assistant in a duration test. Which sound is longer: the first sound, the second sound, or are they the same?", #"Which audio has a higher pitch? The first audio or the second audio?", #"You will listen to an audio clip containing two test tones. Which is higher in pitch?",
            'options_map': {'<A>': "The first sound is longer", '<B>': "The second sound is longer", '<C>': "Both sounds are the same",'<D>': "Unable to determine"},#{'<A>': "The first test tone.", '<B>': "The second test tone.", '<C>': "Unable to determine."}
            'answer_style': answer_style, 
            'loader_function': load_comparison_data_from_disk
        }
    }
    print(f"\nOpt rotations to run: {args.opt_rotate}\n")

    # ========== 主评测循环 ==========
    for seed in args.num_seeds:
        print(f"\n--- RUNNING EVALUATION WITH SEED {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for task_id in args.tasks:
            if task_id not in TASK_CONFIGS:
                print(f"Warning: Task '{task_id}' is not defined. Skipping.")
                continue

            config = TASK_CONFIGS[task_id]
            task_prompt = config['prompt']
            options_map_canonical = config['options_map']
            loader_fn = config['loader_function']

            # 针对每个旋转值分别测试与保存
            for r in args.opt_rotate:
                r = int(r) % 4
                print(f"\n--- Starting Task: {task_id} (rotation={r}) ---")

                # 生成展示用选项映射与列表（A/B/C/D 的位置固定，文本来源旋转）
                rotated_options_map = rotate_options_map(options_map_canonical, r)
                options_list = [rotated_options_map[k] for k in ['<A>', '<B>', '<C>', '<D>']]

                # 加载题目（把展示的 options_list 传给 loader 仅用于拼 query）
                all_task_data = loader_fn(config['trials_dir'], task_id, task_prompt, options_list)
                if not all_task_data:
                    print(f"No data found for task '{task_id}'. Skipping.")
                    continue

                # 输出路径
                output_folder = os.path.join(args.base_results_dir, f'{task_id}/{model_name}')
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f'seed_{seed}_rot{r}_new.jsonl')
                if os.path.exists(output_path):
                    if args.resume:
                        print(f"[resume] Skip existing file: {output_path}")
                        continue  # 直接跳过本 (task, seed, rotation) 组合
                    else:
                        print(f"[overwrite] Remove existing file: {output_path}")
                        os.remove(output_path)

                # 调用对应模型
                current_json_data = all_task_data
                if 'gemini' in model_name or 'gpt' in model_name:
                    evaluation_result = gemini_gpt_process(client, current_json_data, model_name, config['answer_style'], audio_type='wav')
                elif 'qwen-omni' in model_name:
                    evaluation_result = qwen2_5Omni_process(model, processor, current_json_data, config['answer_style'])
                elif 'qwen2-audio' in model_name:
                    evaluation_result = qwen_audio_process(model, processor, model_name, current_json_data, config['answer_style'])
                elif 'kimi' in model_name:
                    evaluation_result = kimi_audio_process(model, current_json_data, config['answer_style'])
                elif 'step' in model_name:
                    evaluation_result = step_audio_process(model, model_name, current_json_data, config['answer_style']) 
                elif  'midasheng' in model_name:
                    evaluation_result = midashenglm_process(model, processor, tokenizer, current_json_data, config['answer_style']) 
                elif  'desta25' in model_name:
                    evaluation_result = desta25_process(model, current_json_data, config['answer_style']) 
                elif 'af3' in model_name:
                    evaluation_result = af3_process(model, model_name, current_json_data, config['answer_style']) 
                elif 'phi4-mm' in model_name:
                    evaluation_result = phi4mm_process( model, generation_config, current_json_data, config['answer_style'])
                elif 'gemma' in model_name:
                    evaluation_result = gemma_process(model, processor, current_json_data, config['answer_style'])
                elif 'salmonn' in model_name:
                    evaluation_result = salmonn_process(model, processor, cfg, current_json_data, config['answer_style'])
                elif 'ming-omni' in model_name:
                    evaluation_result = ming_omni_process(model, processor, current_json_data, config['answer_style'])
                else:
                    raise ValueError(f"Unknown model type: {model_name}")

                # 合并并写出：解析 -> 逆旋转 -> 判分
                cleaned_evaluation_data = []
                for data, prediction in zip(all_task_data, evaluation_result):
                    # 在“展示空间”解析（第三个参数必须是展示映射，用于全文匹配）
                    answer_disp = parse_multi_choice_response(
                        prediction['model_output'], ['A', 'B', 'C', 'D'], rotated_options_map
                    )
                    # 映回 canonical
                    answer_canon = inverse_rotate_letter(answer_disp, r)

                    final_record = data.copy()
                    final_record.update({
                        "prediction_displayed": answer_disp,
                        "prediction_canonical": answer_canon,
                        "model_output": prediction['model_output'],
                        "is_correct": data['answer'] == answer_canon,
                        "seed": seed,
                        "rotation": r,
                        "options_map_displayed": rotated_options_map
                    })
                    final_record.pop('audio', None)
                    cleaned_evaluation_data.append(final_record)

                with open(output_path, 'a') as f:
                    for item in cleaned_evaluation_data:
                        f.write(json.dumps(item) + '\n')

                print(f"Finished Task: {task_id}, rotation={r}. Results appended to: {output_path}")

    print(f"\n{'='*20} PIPELINE FINISHED FOR ALL TASKS, ROTATIONS, AND SEEDS {'='*20}\n")
