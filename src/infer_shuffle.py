#统一通过openai的client
#gemini 使用的是中转 media/
#got4o 用的是官方的
#参考代码：https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/evaluate_chat.py 
#中转 gemini_pro_2.5

#ms-swift 2.5.1 requires transformers<4.47,>=4.33, but you have transformers 4.52.3 which is incompatible.


"""
支持评测模型：
gemini
gpt-4o
qwen2.5 omni
qwen2 Audio
kimi Audio
EchoInk
"""


import json
import os
from tqdm import tqdm
import pdb
import numpy as np
import argparse
import pyarrow.parquet as pq
import io
import re
import sys
import librosa
import base64
import requests
# import httpx
import random
import torch
import soundfile


def enable_proxy1():
    os.environ['http_proxy']='http://closeai-proxy.pjlab.org.cn:23128' 
    os.environ['https_proxy']='http://closeai-proxy.pjlab.org.cn:23128' 
    os.environ['HTTP_PROXY']='http://closeai-proxy.pjlab.org.cn:23128'
    os.environ['HTTPS_PROXY']='http://closeai-proxy.pjlab.org.cn:23128'


'''
media/gemini-2.0-flash
media/gemini-2.0-flash-lite-001
media/gemini-2.5-flash-preview-04-17
media/gemini-2.5-pro-preview-05-06
'''

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
        r"(?i)(?:answer|<answer>|solution|choice|option|correct option is|answer is|答案是|选项是)\s*[:：]*\s*"
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



def build_query(cur_data, answer_style, return_bytes=False, given_caption=False, given_img = False):
    # print('cur_data:', cur_data)
    shuffled, correct_order_str, answer = shuffle_and_get_correct_order()
    # print(shuffled)
    # print(correct_order_str)
    # print(answer)
    cur_data['answer'] = answer+': '+ correct_order_str
    cur_data['shuffled_idx'] = '-'.join([str(c) for c in shuffled])
    audio_inputs = []
    if return_bytes: #对于闭源的api，需要读取bytes
        audio_datas = []
        for k in shuffled:
            with open(cur_data['clip_path'][k - 1], 'rb') as f:
                audio_datas.append(f.read())
        audio_inputs = [base64.b64encode(a).decode('utf-8') for a in audio_datas]
    
    else: #对于开源模型，只需传入audio path
        for k in shuffled:
            audio_inputs.append(cur_data['clip_path'][k-1])
    options = cur_data['options']
    option_text = ""
    for i, Letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G']):
        option_text += f"<{Letter}>: {options[i]}\n"
    question = cur_data['question']
    # question = re.sub(r'\[audio\d+\]', '', question)
    query = question + "\n" + option_text + answer_style 
    
    if given_caption:
        query += f"Here is a caption that describes the full, uncut audio scene to help you reconstruct the original context: {cur_data['global_caption_en']}\n Below are 3 audio clips:\n"
        # f"The following caption describes the entire audio: {cur_data['global_caption_en']}.\n Below are 3 audio clips:\n"
    if given_img:
        query += "The image below illustrates the overall scene corresponding to the audio clips. Based on the scenario depicted in this image, arrange the clips into their most logical chronological order."
    print('shuffled:', shuffled,  ' correct_order_str:', correct_order_str, ' answer:', answer)
    print('query:', query)
    return  query, audio_inputs


def shuffle_and_get_correct_order():
    original = [1, 2, 3]
    shuffled = original.copy()
    random.shuffle(shuffled)
    order_map = {str(shuffled[i-1]): i for i in original}
    correct_order = [order_map[str(k)] for k in original]
    # correct_order_str ='-'.join( [str(c) for c in correct_order])
    correct_order_str =' -> '.join( [f"clip {c}" for c in correct_order])
    for k, v  in options_map.items():
        if v == correct_order_str:
            answer = k
            break
    return shuffled, correct_order_str, answer

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gemini_gpt_process(client, data_inputs, audio_type, model_id, answer_style, given_caption=False, given_img = False): #"gemini-2.5-pro-preview-03-25" "gpt-4o-audio-preview-2024-12-17"  "gemini-2.0-flash"
    result_data = []
    for cur_data in tqdm( data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=True, given_caption=given_caption, given_img=given_img)
        user_content = [
            {
                'type': 'text',
                'text': query
            }
        ]

        if given_img:
            base64_image = encode_image(cur_data['img_path'])
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            )
            user_content.append(
              {"type": "text", "text": "\n Below are 3 audio clips:\n"}  
            )


        if 'gemini' in model_id:
            for idx, audio_data in enumerate(audio_inputs):
                user_content.append({
                    'type': 'text',
                    'text': f'\nclip {idx + 1}:'
                })
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
            for idx, audio_data in enumerate(audio_inputs):
                user_content.append({
                    'type': 'text',
                    'text': f'\nclip {idx + 1}:'
                })
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
                "file_name": cur_data['file_name'],
                'question': query,
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data

def kimi_audio_process(audio_model, data_inputs, answer_style, given_caption=False):
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    #构造query 和 audios
    result_data = []
    for cur_data in tqdm( data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=False, given_caption=given_caption)
        # not support image input
        messages = [
            {"role": "user", "message_type": "text", "content": query}
        ]

        for idx, audio_path in enumerate(audio_inputs):
            messages.append({
                "role": "user", "message_type": "text", 
                "content": f'\nclip {idx + 1}:'
            })
            messages.append({
                "role": "user",
                "message_type": "audio",
                "content":audio_path
            })
        
        print('messages:', messages)


        # messages = [
        #     {"role": "user", "message_type": "text", "content": query},
        #     *[{
        #         "role": "user",
        #         "message_type": "audio",
        #         "content":audio_path
        #                 } for audio_path in audio_inputs]
        # ]

        wav, output = audio_model.generate(messages, **sampling_params, output_type="text")
        print('kimi output:', output)
        result_data.append({
                "file_name": cur_data['file_name'],
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data


#instruction following 能力很差 故放弃测试该模型
def qwen_audio_process(audio_model, processor, model_id, data_inputs, answer_style, given_caption=False):
    result_data = []
    for cur_data in tqdm(data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=False, given_caption=given_caption)
        audio_data = [librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)[0] for audio_path in audio_inputs]
        
        if 'instruct' in model_id:
            user_content = [
                {
                    'type': 'text',
                    'text': query
                }
            ]
            for idx, audio_path in enumerate(audio_inputs):
                # user_content.append({
                #     'type': 'text',
                #     'text': f'\nclip {idx + 1}:'
                # })
                user_content.append({
                    'type': 'audio',
                    'audio_url': audio_path
                })
            conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}, #Audio Analysis Inference
                            {'role': 'user',
                             'content': user_content}]
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )                             
        else:
            print('not support!!!')
            return
            # text = '<|audio_bos|><|AUDIO|><|audio_eos|>' + query
        
        print("text:", text) #qwen2-audio-instruct 会自动在audio前面加上标签 “Audio 1:” 故在prompt中去掉 “clip i”的标注
        '''
        You are a specialized sound event ordering expert. Please listen to the following three audio clips labeled clip 1, clip 2, and clip 3, and determine the most natural chronological order in which these sounds would typically occur in the real world. 
        <A>: clip 1 -> clip 2 -> clip 3
        <B>: clip 1 -> clip 3 -> clip 2
        <C>: clip 2 -> clip 1 -> clip 3
        <D>: clip 2 -> clip 3 -> clip 1
        <E>: clip 3 -> clip 1 -> clip 2
        <F>: clip 3 -> clip 2 -> clip 1
        <G>: I don't know.
        Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
        Audio 2: <|audio_bos|><|AUDIO|><|audio_eos|>
        Audio 3: <|audio_bos|><|AUDIO|><|audio_eos|>
        <|im_end|>
        <|im_start|>assistant

        '''
        inputs = processor(
            text=text,
            audio=audio_data, #https://github.com/QwenLM/Qwen2-Audio/issues/146
            return_tensors='pt',
            padding=True,
            sampling_rate=processor.feature_extractor.sampling_rate
        )
        inputs = inputs.to(audio_model.device).to(audio_model.dtype)
        
        generate_ids = audio_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        output = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print('qwen2-audio output:', output)

        result_data.append({
                "file_name": cur_data['file_name'],
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data

def step_audio_process(audio_model, model_id, data_inputs, answer_style, given_caption=False):
    result_data = []
    for cur_data in tqdm(data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=False, given_caption=given_caption)
        
        user_content = [
            {
                'type': 'text',
                'text':  query   #'descripe the audio first.' #
            }
        ]
        for idx, audio_path in enumerate(audio_inputs):
            user_content.append({
                'type': 'text',
                'text': f'\nclip {idx + 1}:'
            })
            user_content.append({
                'type': 'audio',
                'audio': audio_path
            })
            

        if 'base' not in model_id:
            #https://github.com/stepfun-ai/Step-Audio2/blob/main/examples.py#L148
            messages = [
                {"role": "system", "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."},
                {"role": "human", "content": user_content},
                {"role": "assistant", "content": None}
            ]
        else:
            pass #text = '<|audio_bos|><|AUDIO|><|audio_eos|>' + query
        
        print("messages:", messages)
        tokens, output, _ = audio_model(messages, max_new_tokens=256, num_beams=2)

        print('【step-audio-2 output】:', output)

        result_data.append({
                "file_name": cur_data['file_name'],
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data


def qwen2_5Omni_process(audio_model, processor, data_inputs, answer_style, USE_AUDIO_IN_VIDEO=False, given_caption=False, given_img=False):
    #构造query 和 audios
    result_data = []
    for cur_data in tqdm( data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=False, given_caption=given_caption, given_img=given_img)
        user_content = [
            {
                'type': 'text',
                'text': query
            }
        ]
        if given_img:
            user_content.append(
                {
                    "type": "image",
                    "image": cur_data['img_path']
                }
            )
            user_content.append(
              {"type": "text", "text": "\n Below are 3 audio clips:\n"}  
            )


        for idx, audio_path in enumerate(audio_inputs):
            user_content.append({
                'type': 'text',
                'text': f'\nclip {idx + 1}:'
            })
            user_content.append({
                'type': 'audio',
                'audio': audio_path
            })

        messages = [
            # {
            #     "role": "system",
            #     "content": [
            #         {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            #     ],
            # },#加了这个就变智障了
            {
                'role': 'user',
                'content': user_content
            }
        ]
        print('message:', messages)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        print('text:', text[0])
        audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(audio_model.device).to(audio_model.dtype)
        #text_ids, audio = audio_model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        #不生成audio 下面两行会产生很多废话 https://github.com/QwenLM/Qwen2.5-Omni/issues/312
        # text_ids = audio_model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False, thinker_do_sample=False)
        # text_ids = text_ids[:, inputs.input_ids.size(1):]
        # output = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        with torch.no_grad():
            model_output = audio_model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                thinker_max_new_tokens=1024,
                return_dict_in_generate=True,
                return_audio=False,
                thinker_do_sample=False,
                # repetition_penalty=1.0
            )
            output = processor.batch_decode(
                model_output.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        # model_output = audio_model.generate(
        #     **inputs,
        #     max_new_tokens=512,#120 256
        #     eos_token_id=processor.tokenizer.eos_token_id,
        #     do_sample=True,
        #     output_logits=True,
        #     return_dict_in_generate=True,
        #     temperature=0.01,
        # )

        # output = processor.batch_decode(
        #     model_output.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]
        
        print('omni output:', output)
        result_data.append({
                "file_name": cur_data['file_name'],
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data


def ming_omni_process(audio_model, processor, data_inputs, answer_style, USE_AUDIO_IN_VIDEO=False, given_caption=False, given_img=False):
    #构造query 和 audios
    result_data = []
    for cur_data in tqdm( data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=False, given_caption=given_caption, given_img=given_img)
        user_content = [
            {
                'type': 'text',
                'text': query
            }
        ]
        if given_img:
            user_content.append(
                {
                    "type": "image",
                    "image": cur_data['img_path']
                }
            )
            user_content.append(
              {"type": "text", "text": "\n Below are 3 audio clips:\n"}  
            )


        for idx, audio_path in enumerate(audio_inputs):
            user_content.append({
                'type': 'text',
                'text': f'\nclip {idx + 1}:'
            })
            user_content.append({
                'type': 'audio',
                'audio': audio_path
            })

        messages = [
            {
                'role': 'HUMAN',
                'content': user_content
            }
        ]
        print('message:', messages)
        #https://github.com/inclusionAI/Ming/blob/main/cookbook.ipynb 详见ASR
        output = ming_omni_generate(messages=messages, processor=processor, model=audio_model)
        
        print('Ming omni output:', output)
        result_data.append({
                "file_name": cur_data['file_name'],
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data

#TODO 待确认是否支持多音频输入
def midashenglm_process(audio_model, processor, tokenizer, data_inputs, answer_style, given_caption=False):
    result_data = []
    for cur_data in tqdm(data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=False, given_caption=given_caption)

        user_content = [
            {
                'type': 'text',
                'text': query
            }
        ]
        for idx, audio_path in enumerate(audio_inputs):
            user_content.append({
                'type': 'text',
                'text': f'\nclip {idx + 1}:'
            })
            user_content.append({
                'type': 'audio',
                'path': audio_path
            })

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
                    "content": user_content,
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
                "file_name": cur_data['file_name'],
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data

def desta25_process(audio_model, data_inputs, answer_style, given_caption=False):
    result_data = []
    for cur_data in tqdm(data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=False, given_caption=given_caption)

        # https://github.com/kehanlu/DeSTA2.5-Audio
        messages = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions."
            },
            {
                "role": "user",
                "content": query + '\nclip 1: <|AUDIO|>\nclip 2: <|AUDIO|>\nclip 3: <|AUDIO|>\n',
                "audios": [{
                    "audio": audio_path,  # Path to your audio file
                    "text": None
                } for audio_path in audio_inputs]
            }
        ]
        print('messages:', messages)
        with torch.no_grad():
            outputs = audio_model.generate(
                messages=messages,
                do_sample=False,
                top_p=1.0,
                temperature=1.0,
                max_new_tokens=1024
            )
            output = outputs.text[0]

        print('desta25 output:', output)

        result_data.append({
                "file_name": cur_data['file_name'],
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data

def phi4mm_process(audio_model, generation_config, data_inputs, answer_style, given_caption=False, given_img=False):
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    result_data = []
    for cur_data in tqdm(data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=False, given_caption=given_caption, given_img=given_img)

        # https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_inference_phi4mm.py#L162
        prompt = f'{user_prompt}{query}'
        if given_img:
            print('not support yet')
        prompt += f'\nclip 1: <|audio_1|>\nclip 2: <|audio_2|>\nclip 3: <|audio_3|>\n{prompt_suffix}{assistant_prompt}' 
        print('prompt:', prompt)
        audios = [soundfile.read(audio_path) for audio_path in audio_inputs]
        inputs = processor(text=prompt, audios=audios, return_tensors='pt').to(audio_model.device)
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
                "file_name": cur_data['file_name'],
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data

def gemma_process(audio_model, processor, data_inputs,answer_style, given_caption=False, given_img=False):
    result_data = []
    for cur_data in tqdm(data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=False, given_caption=given_caption, given_img=given_img)

        user_content = [
            {
                'type': 'text',
                'text': query
            }
        ]
        for idx, audio_path in enumerate(audio_inputs):
            user_content.append({
                'type': 'text',
                'text': f'\nclip {idx + 1}:'
            })
            user_content.append({
                'type': 'audio',
                'audio': audio_path
            })
        #https://huggingface.co/google/gemma-3n-E4B-it
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        print('messages:', messages )

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(audio_model.device).to(audio_model.dtype)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = audio_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generation = generation[0][input_len:]

        output = processor.decode(generation, skip_special_tokens=True)

        
        print('gemma output:', output)

        result_data.append({
                "file_name": cur_data['file_name'],
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data

def af3_process(audio_model, model_id, data_inputs, answer_style, given_caption=False):
    result_data = []
    for cur_data in tqdm(data_inputs):
        query, audio_inputs = build_query(cur_data, answer_style, return_bytes=False, given_caption=given_caption)

        if 'think' in model_id:
            query += ' Please think and reason about the input audio before you respond.' 

        #https://github.com/NVIDIA/audio-flamingo/blob/audio_flamingo_3/llava/cli/infer_audio.py#L79
        prompts = [query]
        for idx, audio_path in enumerate(audio_inputs):
            prompts.append( f'\nclip {idx + 1}:')
            prompts.append( Sound(audio_path))

        print('messages:', prompts)
        output = model.generate_content(prompts, response_format=None)
        print('af3 output:', output)
        result_data.append({
                "file_name": cur_data['file_name'],
                "category": cur_data['new_category'],
                'shuffled_idx': cur_data['shuffled_idx'],
                "model_output": output,
                "answer": cur_data['answer']
            })
    return result_data
#model:
#gpt-4o-audio-preview-2025-06-03
#gpt-4o-audio-preview-2024-12-17
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--model',default="gemini-2.5-pro", type=str) #"media/gemini-2.5-pro-preview-05-06"  "gemini-2.0-flash-001"  "gemini-2.0-flash" "gemini-2.0-flash" "gemini-2.5-pro-preview-05-06"   "gemini-2.5-pro-preview-03-25" 不能access audio 为什么啊？？  
    # parser.add_argument('--dataset', type=str, default='AV-Odyssey/Deaftest_dataset')
    parser.add_argument('--given_caption', action='store_true', help='Set to True if a global caption describing the full event is provided; otherwise, it defaults to False.')
    parser.add_argument('--given_img', action='store_true', help='Set to True if an image describing the full event is provided; otherwise, it defaults to False.')
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--input_jsonl_path', type=str, default='/mnt/shared-storage-user/liuzihan/ST_Benchmark/AudioOrderTest/data_samples/cleaned_total_900_en_cap_picked.jsonl') #'/mnt/shared-storage-user/liuzihan/ST_Benchmark/AudioOrderTest/data_samples/cleaned_total_0909.jsonl'
    args = parser.parse_args()
    model_name = args.model
    input_jsonl_path = args.input_jsonl_path
    audio_root = '/mnt/shared-storage-user/liuzihan/ST_Benchmark/AudioOrderTest/data_samples/audios'
    seed = args.seed
    random.seed(args.seed)


        
    output_folder = f'./test_results_after0909'
    # none 表示answer stype为空
    if args.given_caption:
        output_path = os.path.join(output_folder+'_given_caption', model_name, f"seed{seed}_none_new.jsonl")
    elif args.given_img:
        output_path = os.path.join(output_folder+'_given_img', model_name, f"seed{seed}_none_new.jsonl")
    else:
        output_path = os.path.join(output_folder, model_name, f"seed{seed}_none_new.jsonl") #new表示用了修正后的parse函数
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    

    # task_prompt = "You are a specialized sound event ordering expert. Please listen to the following three audio clips labeled 1, 2, and 3, and determine the most natural chronological order in which these sounds would typically occur in the real world. First, explain your reasoning. Then, choose the correct sequence in the format: N-N-N. "
    
    # task_prompt = "You are a specialized sound event ordering expert. Please listen to the following three audio clips labeled clip 1, clip 2, and clip 3, and determine the most natural chronological order in which these sounds would typically occur in the real world. First, explain your reasoning. Then, choose the correct sequence:"# in the format: clip N -> clip N - clip N.
    task_prompt = "You are a specialized sound event ordering expert. Please listen to the following three audio clips labeled clip 1, clip 2, and clip 3, and determine the most natural chronological order in which these sounds would typically occur in the real world. "
    
    answer_style = ''
    # answer_style = 'First, describe the content of each clip in detail. Then, explain your reasoning. Finally, present your answer.' +"Answer using the exact format <X>, where X is the letter corresponding to your selected option (e.g., <A>).\n"#"Answer with the option's letter from the given choices." # directly
    


    # base_url = "http://35.220.164.252:3888/v1"
    # if 'gemini' in model_name:
    #     api_key = "sk-QUYe4Wtk4MLS7GLnkCjGjKqZk4iwb0V8OQPWP4UcuykIREHS"
    # else:
    #     api_key = "sk-TL4hrsr2r1XhIXJp0kibaqQFY73JViYAgErUZn4WapTZEaKm"
    if 'gemini' in model_name or 'gpt' in model_name:
        from openai import OpenAI
        base_url = "https://api.boyuerichdata.opensphereai.com/v1" #"http://35.220.164.252:3888/v1" # 'http://127.0.0.1:8005/v1/'  #
        api_key = "sk-TL4hrsr2r1XhIXJp0kibaqQFY73JViYAgErUZn4WapTZEaKm" 
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    # elif 'gpt' in model_name:
    #     from openai import OpenAI
    #     enable_proxy1()
    #     api_key = "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"
    #     client = OpenAI(api_key=api_key)
    
    elif "qwen2-audio" in model_name:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        if 'instruct' in model_name:
            model_path = "/mnt/shared-storage-user/liuzihan/saved_ckpts/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a" #"Qwen/Qwen2-Audio-7B-Instruct"
        else:
            model_path = "/mnt/shared-storage-user/liuzihan/saved_ckpts/models--Qwen--Qwen2-Audio-7B/snapshots/dd84470756e6277a71d4d7188773a43cde92696e" #'Qwen/Qwen2-Audio-7B'
        model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map='cuda').eval()
        processor = AutoProcessor.from_pretrained(model_path)
    elif 'qwen-omni' in model_name or 'EchoInk' in model_name:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info
        # model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-Omni-7B",
        #     torch_dtype="auto",
        #     device_map="auto",
        #     attn_implementation="flash_attention_2",
        # )
        if 'qwen-omni' in model_name:
            model_path = "/mnt/shared-storage-user/liuzihan/saved_ckpts/Qwen2.5-Omni-7B"  #"Qwen/Qwen2.5-Omni-7B"
        elif 'EchoInk' in model_name:
            model_path = "harryhsing/EchoInk-R1-7B"
            QUESTION_TEMPLATE = (
                "{Question}\n"
                "Please think about this question as if you were a human pondering deeply. "
                "Make sure to carefully consider both the visual and audio information before answering. "
                "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
                "It's encouraged to include self-reflection or verification in the reasoning process. "
                "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
            )
            task_prompt = QUESTION_TEMPLATE.format(Question=task_prompt)
            answer_style = ''

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2").eval() 
        # audio_model = Qwen2_5OmniForConditionalGeneration.from_pretrained( model_path, torch_dtype="auto", device_map="cuda").eval() #"auto"
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        model.disable_talker()
    
    #kimi 不支持多audio输入 之后可尝试将多个clip 拼接起来 中间用育婴"clip X" 进行连接
    elif 'kimi' in model_name:
        #虚拟环境换成 kimi 原来的 transformers 4.51.3； 由于适配ming-omni 换成了 transformers==4.52.4
        kimi_path =  "/mnt/shared-storage-user/liuzihan/Kimi-Audio"  #
        sys.path.append(kimi_path)
        from kimia_infer.api.kimia import KimiAudio
        model_path = "/mnt/shared-storage-user/liuzihan/saved_ckpts/huoshan_cache/huggingface/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b" #'/mnt/shared-storage-user/liuzihan/saved_ckpts/models--moonshotai--Kimi-Audio-7B/snapshots/d90be7113cd5be6bcf54fae2aabcad060a23f6cf'
        model = KimiAudio(
            model_path=model_path,
            load_detokenizer=False, #True,
        ) #.to('cuda').eval()
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
        processor = AutoProcessor.from_pretrained(ming_path, trust_remote_code=True, local_files_only=True) #ming_path
    
    
    elif 'baichuan-omni' in model_name: #代码不清楚
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model.training = False
        model.bind_processor(tokenizer, training=False, relative_path="/")


    options_map = {
        '<A>':'clip 1 -> clip 2 -> clip 3',
        '<B>':'clip 1 -> clip 3 -> clip 2',
        '<C>':'clip 2 -> clip 1 -> clip 3',
        '<D>':'clip 2 -> clip 3 -> clip 1',
        '<E>':'clip 3 -> clip 1 -> clip 2',
        '<F>':'clip 3 -> clip 2 -> clip 1',
        '<G>':"I don't know."
    }

    #统计outpu_path中已经处理完的file_name
    processed_files = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_files.add(data['file_name'])
                


    total_questions = []

    with open(input_jsonl_path , "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data['file_name'] in processed_files:
                # print(f"Skipping already processed file: {data['file_name']}")
                continue
            if data['clips_len'] < 3: #先跳过两段的
                continue
            data['clip_path'] = [os.path.join(audio_root, data['src'], data["file_name"].split('.wav')[0], f"{clip_idx}.wav") for clip_idx in ["clip_1", "clip_2", "clip_3"]]
            if args.given_img:
                data['img_path'] = os.path.join(img_dir, data["file_name"].split('.wav')[0]+'.png') 
            data['question'] = task_prompt
            data['options'] = [options_map[i] for i in ['<A>', '<B>', '<C>', '<D>', '<E>', '<F>', '<G>']]
            total_questions.append(data)
    # total_questions = total_questions[:5]
    print(f"Total questions loaded: {len(total_questions)}")

    



    all_evaluation_results = []
    batch_size  = 10
    for current_question_id in range(0, len(total_questions), batch_size):
        current_json_data = total_questions[current_question_id: min(current_question_id + batch_size, len(total_questions))]
        if 'gemini' in model_name or 'gpt' in model_name:
            evaluation_result = gemini_gpt_process(client, current_json_data, audio_type='wav', model_id=model_name, answer_style=answer_style, given_caption=args.given_caption, given_img = args.given_img)
        elif 'qwen2-audio' in model_name:
            evaluation_result = qwen_audio_process(model, processor, model_name, current_json_data, answer_style=answer_style, given_caption=args.given_caption) #msswift_qwen2_audio_process(audio_model, audio_template, current_json_data, answer_style=answer_style, given_caption=args.given_caption)
        elif 'qwen-omni' in model_name  or 'EchoInk' in model_name:
            evaluation_result = qwen2_5Omni_process(model, processor, current_json_data, answer_style=answer_style, given_caption=args.given_caption, given_img=args.given_img)
        elif 'kimi' in model_name:
            evaluation_result = kimi_audio_process(model, current_json_data, answer_style=answer_style, given_caption=args.given_caption)
        elif 'step' in model_name:
            evaluation_result = step_audio_process(model, model_name, current_json_data,answer_style=answer_style, given_caption=args.given_caption)
        elif 'midasheng' in model_name:
            evaluation_result =midashenglm_process(model, processor, tokenizer, current_json_data, answer_style=answer_style, given_caption=args.given_caption)
        elif  'desta25' in model_name:
            evaluation_result = desta25_process(model, current_json_data, answer_style=answer_style, given_caption=args.given_caption) 
        elif 'af3' in model_name:
            evaluation_result = af3_process(model, model_name, current_json_data, answer_style=answer_style, given_caption=args.given_caption) 
        elif 'phi4-mm' in model_name:
            evaluation_result = phi4mm_process( model, generation_config, current_json_data, answer_style=answer_style, given_caption=args.given_caption)
        elif 'gemma' in model_name:
            evaluation_result = gemma_process(model, processor, current_json_data, answer_style=answer_style, given_caption=args.given_caption)
        elif 'ming-omni' in model_name:
            evaluation_result = ming_omni_process(model, processor, current_json_data, answer_style=answer_style, given_caption=args.given_caption)

        # elif 'salmonn' in model_name:
        #     evaluation_result = salmonn_process(model, processor, cfg, current_json_data, answer_style=answer_style, given_caption=args.given_caption)
        
        else:
            raise ValueError(f"Unknown model type: {model_name}")   



        # clean the answer, following MMMU (https://github.com/MMMU-Benchmark/MMMU)
        cleaned_evaluation_data = []
        for data, prediction in zip(current_json_data, evaluation_result):
            option_list = options_map
            answer = parse_multi_choice_response(prediction['model_output'], ['<A>', '<B>', '<C>', '<D>', '<E>', '<F>', '<G>'], option_list)
            prediction['prediction'] = answer
            prediction["is_correct"] = prediction['answer'].startswith(answer)
            cleaned_evaluation_data.append(prediction)

        all_evaluation_results = all_evaluation_results + cleaned_evaluation_data

        

        with open(output_path, 'a', encoding="utf-8") as f:
            for item in cleaned_evaluation_data:
                # item.pop("answer")
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        
        
        
        
    