import random
import pandas as pd


class MCQBaseDataset(Dataset):
    DATASET_NAME = None
    def __init__(self, dataset_root=''):
        assert self.DATASET_NAME is not None
        self.dataset_file = os.path.join(self.dataset_root, f'{self.DATASET_NAME}.jsonl')
        self.data = self.load_data(self.dataset_file)

        #加载数据集

    def set_demo_mode(self, demo=True):
        self.demo = demo
        self.data = self.data[:10]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.build_prompt(idx)
    
    def load_data(self, dataset):
        pass


    #构建msg {'meta': {}, 'prompts': [{'text': text_query}, {'audio': audio_path}] }
    # line {"split": ['perception_range｜sensitivity', 'holistic_temporal|spatial'], "task_sub": ['range','sensitivity', 'temporal', 'spatial']  , "audio_path": str | list(str), "question": "What is the motion trajectory of a horse-drawn carriage?", "options": ["From left to right", "From right to left", "Remains unchanged"], "answer": "From right to left", "question_categpry": "dynamic trajectory tracking"}
    # ?? 是不是对于所有模型的 audio都是放在前面的？
    # 对于排序任务：
    # answer 是根据事实的shuffle来定的 
    #ine {"split": ['perception_range｜sensitivity', 'holistic_temporal|spatial'], "task_sub": ['range','sensitivity', 'temporal', 'spatial']  , "audio_path": str | list(str), "question": "What is the motion trajectory of a horse-drawn carriage?", "options": ["From left to right", "From right to left", "Remains unchanged"], "question_categpry": "dynamic trajectory tracking"}

    def build_prompt(self, idx: int | str) -> dict: 
        





    #line Meta:{task-meta, task_sub, question-type, idx, options:[], answer:<X>, rotate_id, shuffle_seed, prompts }
    def evaluate(self, eval_file, dump_judge=True):
        """
        evaluate performance based on result jsonl file.
        if dump_judge=True, will dump the judge result to this file.

        The jduge result will be a copy of eval_file, with additional fields: judge_result
        """
        df = pd.read_json(eval_file, lines=True)
        metrics = {}
        judge_results = {}
        for task, group in df.groupby('question_categpry'):



        if dump_judge:
            # dump the judge result to the eval_file
            all_df = []
            for task, judge_result in judge_results.items():
                df = pd.DataFrame(judge_result)
                all_df.append(df)
            all_df = pd.concat(all_df)
            save_file = eval_file.replace(
                '.jsonl', f'_{judge_model_name}_judge.jsonl')
            all_df.to_json(save_file, orient='records',
                           lines=True, force_ascii=False)
        return result
    

    # build prompting 后给的options 以及 answer要是 rotate或者shuffle好的
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

