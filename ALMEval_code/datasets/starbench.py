import os
import json
import re
import string
import pandas as pd
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from .utils import binaural2single
import random
from loguru import logger
import numpy as np

class MCQBaseDataset(Dataset):
    DATASET_ALIAS = None
    JSON_PATH = None
    def __init__(self, dataset_root):
        assert self.DATASET_ALIAS, "Subclass must define DATASET_ALIAS"
        assert self.JSON_PATH, "Subclass must define JSON_PATH"

        self.dataset_root = dataset_root
        self.dataset_file = os.path.join(self.dataset_root, self.JSON_PATH)
        self.data = self.load_data(self.dataset_file)
        self.demo = False


    def set_demo_mode(self, demo=True, limits=10):
        self.demo = demo
        if demo:
            n = len(self.data)
            if n <= limits:
                self.data = self.data[:]  
            else:
                indices = np.linspace(0, n - 1, limits, dtype=int)
                self.data = [self.data[i] for i in indices]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the raw data item. The prompt is built in run.py.
        """
        return self.data[idx]
    
    def load_data(self, dataset_file):
        """Loads data from a single JSON file containing a list of dicts."""
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    
    # 默认的方法 即不改变数据原本的options audio_paths， 抽取answer letter
    def _prepare_options_and_audios(self, line: dict, **kwargs) -> tuple:
        options = line['options']
        audio_paths = line["audio_paths"]
        answer = line.get("answer", "MISSING ANSWER VALUE")
        assert answer in options, f"answer not in options! answer:{answer}, options:{options}"
        ans_idx = options.index(answer)
        answer_letter = '<'+string.ascii_uppercase[ans_idx]+'>' 
        return  options, audio_paths, answer_letter
    
    def _build_audio_prompts(self, audio_paths: list) -> list:
        """
        [DEFAULT IMPLEMENTATION in Base Class]
        Builds audio prompts. This version is used by all subclasses that
        do not provide their own override. It handles one or multiple standard audio files.
        """
        prompts = []
        if not audio_paths: 
            return prompts
        
        if len(audio_paths) > 1:
            for i, path in enumerate(audio_paths):
                prompts.extend([
                    {'type': 'text', 'value': f'Audio {i+1}:'},
                    {'type': 'audio', 'value': os.path.join(self.dataset_root, path)}
                ])
        else: # Handle single audio file without "Audio 1:" prefix
            prompts.append({'type': 'audio', 'value': os.path.join(self.dataset_root, audio_paths[0])})
        return prompts


    def build_prompt(self, line: int | dict , **kwargs) -> dict:
        """Construct a user conversation from a data source. Currently, we only consider single-turn conversations.
        return msg:
        msg = {
            "meta": {
                "id": ...,
                "task": ...,
                "category": ...,
                "sub-category": ...,
                "options": ...,
                "answer": ...,
                "answer_letter": ...,
                "rotate_id": ...,
            },
            "prompts": [
                {"type": "text", "value": "xxxx"},
                {"type": "audio", "value": "audio1.wav"},
                {"type": "text", "value": "xxxx"},
                {"type": "audio", "value": "audio2.wav"},
                ...
            ]
        }  
        """ 
        if isinstance(line, int):
            line = self.data[line]

        question = line['question']
        options, audio_paths, answer_letter = self._prepare_options_and_audios(line, **kwargs)
        #'Options:\n' +
        options_prompt_str =  ''.join(
            f"<{string.ascii_uppercase[i]}>: {opt}\n" for i, opt in enumerate(options)
        )
        
        final_text_prompt = f"{question}\n{options_prompt_str}"

        # Construct prompts list
        prompts= []
        prompts.extend(self._build_audio_prompts(audio_paths))
        prompts.append({'type':'text', 'value':final_text_prompt})

        msg = {
            "meta":{
                "id": line['id'],
                "task": line['task'],
                "category": line.get('category', None),
                "sub-category": line.get('sub-category', None),
                "options": options,
                "answer": line['answer'] ,
                "answer_letter": answer_letter,
                "rotate_id": kwargs.get('rotate_id', 0),
                # "unique_id": f"{line['id']}@{kwargs.get('rotate_id', 0)}"
            },
            "prompts":prompts
        }
        return msg

    def evaluate_item(self, item: dict, model_response: str) -> dict:
        prediction_letter = self.parse_multi_choice_response(model_response, item['options'])
        result_info = item.copy()
        result_info.update({
            'prediction': prediction_letter,
            'is_correct': (prediction_letter == item['answer_letter']),
            'model_response': model_response,
        })
        return result_info
    
    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Helper to compute AA and ACR for a given DataFrame."""
        if df.empty: return {"AA": "0.00%", "ACR": "0.00%", "count": 0}
        aa = df['is_correct'].mean() * 100
        grouped = df.groupby('id')['is_correct'].all()
        acr = grouped.mean() * 100
        return {"AA": f"{aa:.2f}%", "ACR": f"{acr:.2f}%", "total_runs": len(df), "unique_samples": df['id'].nunique()}
    
    
    def evaluate(self, eval_file: str, robust_eval=True, reeval=False) -> dict:
        """
        Calculates final performance metrics (Total, by Category, by Sub-category)
        from a results file and returns a clean, serializable dictionary.
        """
        if not os.path.exists(eval_file) or os.path.getsize(eval_file) == 0:
            return {"error": f"Evaluation file not found or is empty: {eval_file}"}
        
        if reeval:
            tmp_path = eval_file + ".tmp"
            updated = 0
            total = 0
            # Re-evaluate each line and rewrite the source file
            with open(eval_file, "r", encoding="utf-8") as fin, \
                open(tmp_path,  "w", encoding="utf-8") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        item = json.loads(line)
                        model_resp = item.get("model_response", "")
                        # Re-evaluate this record
                        new_item = self.evaluate_item(item, model_resp)
                        fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                        updated += 1
                    except Exception as e:
                        fout.write(line + "\n")
           
            os.replace(tmp_path, eval_file)
            logger.info(f"Re-evaluated {updated}/{total} items and rewrote {eval_file}.")
           
        df = pd.read_json(eval_file, lines=True)

        logger.info(f"Verifying the completeness of test samples in {eval_file}")
        if robust_eval:
            robustness_runs = 3
        else:
            robustness_runs = 1 
        expected_total = len(self.data) * robustness_runs
        assert len(df) == expected_total, \
            f"Row count mismatch: expected {len(self.data)} * {robustness_runs} = {expected_total}, got {len(df)}"
        counts = df['id'].value_counts()
        assert all(counts == robustness_runs), \
            f"Some ids do not have exactly {robustness_runs} runs"

        results = {"Total": self._calculate_metrics(df)}

        if 'category' in df.columns and df['category'].notna().any():
            # Convert keys to string to ensure JSON serializability
            results['By Category'] = {
                str(cat): self._calculate_metrics(group) for cat, group in df.groupby('category')
            }
        
        if 'sub-category' in df.columns and df['sub-category'].notna().any():
            results['By Sub-category'] = {
                str(sub_cat): self._calculate_metrics(group) for sub_cat, group in df.groupby('sub-category')
            }
        return results

    
    def parse_multi_choice_response(self, response: str, options: list) -> str:
        """
        Parse the model's response and extract the predicted multiple-choice letter.
        Returns a string like '<A>' or 'Z' (for no valid match).
        """
        letter2options ={}
        for i in range(len(options)):
            l = string.ascii_uppercase[i]
            letter2options[l] = options[i]

        all_choices = list(letter2options.keys())
        choices_str = "".join(all_choices)

        # --- Priority 1: find the last explicitly mentioned choice after keywords ---
        # e.g., "Answer: (A)", "The correct option is B", "答案是C"
        keyword_matches = []
        keyword_pattern = re.compile(
            r"(?i)(?:answer|<answer>|solution|choice|option|correct option is|answer is|答案是|选项是)\s*[:：]*\s*"
            r"(?:"
            r"[\(\<\[\{{]\s*([{choices}])\s*[\)\]\>\}}]"  # pattern like (A), <A>, [A], {A}
            r"|\b([{choices}])\b"  # standalone letter A
            r")".format(choices=choices_str)
        )

        for match in re.finditer(keyword_pattern, response):
            found_choice_first = match.group(1) 
            found_choice_second = match.group(2)
            if found_choice_second:
                keyword_matches.append((found_choice_second, match.start(), 0)) # higher priority for bracketed forms like (A)
            if found_choice_first:
                keyword_matches.append((found_choice_first, match.start(), 1))
                
        if keyword_matches:
            # sort by type and position, return the last match
            keyword_matches.sort(key=lambda x: (x[2], x[1]))
            return '<'+keyword_matches[-1][0]+'>'

        # --- Priority 2: check general patterns or option text matches ---
        standalone_matches = []
        standalone_pattern = re.compile(
            r"[\(\<\[\{{]\s*([{choices}])\s*[\)\]\>\}}]" # pattern like (A), <A>, [A], {A}
            r"|\b([{choices}])\b".format(choices=choices_str) # standalone letter A
        )

        for match in re.finditer(standalone_pattern, response):
            found_choice_first = match.group(1)  
            found_choice_second = match.group(2)
            if found_choice_first:
                standalone_matches.append(('<'+found_choice_first+'>', match.start(), 2))
            if found_choice_second:
                standalone_matches.append(('<'+found_choice_second+'>', match.start(), 0))  
        # also match by full option text    
        if len(response.split()) > 2:
            for choice_letter, choice_text in letter2options.items():
                pattern = re.compile(rf'(?<![A-Za-z]){re.escape(choice_text)}(?![A-Za-z])', re.IGNORECASE)
                for match in re.finditer(pattern, response):
                    standalone_matches.append(('<'+choice_letter+'>', match.start(), 1))

        if standalone_matches:
            # sort by priority and position, return the last valid choice
            standalone_matches.sort(key=lambda x: (x[2], x[1])) 
            return standalone_matches[-1][0]

        # fallback: no valid match found
        return 'Z'



class TemporalReasoningDataset(MCQBaseDataset):
    DATASET_ALIAS="tr"
    JSON_PATH="meta_info/holistic_reasoning_temporal.json"
    
    def _perm_by_roundrobin(self, rotate_id: int, tid: str) -> list[int]:
        PERMS_123 = [
            [1,2,3],
            [1,3,2],
            [2,1,3],
            [2,3,1],
            [3,1,2],
            [3,2,1],
        ]
        SUB_TASKS = ["OSM", "ISSE", "TAO", "DSS", "ETC"]
        parts = tid.split('_')
        subc  = parts[-2]
        subid = int(parts[-1])

        offset = (rotate_id % 6 + SUB_TASKS.index(subc)) % 6
        idx = (subid + offset) % 6
        return PERMS_123[idx]

    def _prepare_options_and_audios(self, line: dict, rotate_id: int = 0, **kwargs) -> tuple:
        """shuffles audio paths and determines the
        correct answer from the list of text options.
        """
        options = line['options']
        original = [1, 2, 3]
        shuffled = self._perm_by_roundrobin(rotate_id, line['id'])
        order_map = {str(shuffled[i-1]): i for i in original}
        correct_order = [order_map[str(k)] for k in original]
        answer =' -> '.join( [f"clip {c}" for c in correct_order])
        shuffled_seg_order = '-'.join([str(c) for c in shuffled])
        audio_paths = []
        for k in shuffled:
            audio_paths.append(line['audio_paths'][k-1])
        ans_idx = options.index(answer)    
        answer_letter = '<'+string.ascii_uppercase[ans_idx]+'>' 
        return options, audio_paths, answer, answer_letter, shuffled_seg_order

    def _build_auxiliary_prompts(self, line: dict)-> list:
        """A hook for subclasses to add aluxiliary prompts."""
        return []

    def build_prompt(self, line: int | dict, **kwargs) -> dict:
        if isinstance(line, int):
            line = self.data[line]
        
        options, audio_paths, answer, answer_letter, shuffled_seg_order = self._prepare_options_and_audios(line, **kwargs)
        #'Options:\n' + 
        options_prompt_str = ''.join(
            f"<{string.ascii_uppercase[i]}>: {opt}\n" for i, opt in enumerate(options)
        )

        question = line['question']
        final_text_prompt = f"{question}\n{options_prompt_str}"

        #construct prompts
        prompts= []
        prompts.append(
            {'type':'text', 'value':final_text_prompt},
        )
        prompts.extend(self._build_auxiliary_prompts(line))

        assert len(audio_paths) == 3, "Temporal reasoning expects 3 audio clips."
        for i in range(len(audio_paths)):
            prompts.extend(
                [
                    {'type':'text', 'value':f'\nclip {i+1}:'},
                    {'type':'audio', 'value': os.path.join(self.dataset_root, audio_paths[i])}
                ]
            ) 
        msg = {
            "meta":{
                "id": line['id'],
                "task": line['task'],
                "category": line.get('category', None),
                "sub-category": line.get('sub-category', None),
                "options": options,
                "answer": answer ,
                "answer_letter": answer_letter,
                "shuffled_seg_order":shuffled_seg_order,
                "rotate_id": kwargs.get('rotate_id', 0),
                # "unique_id": f"{line['id']}@{kwargs.get('rotate_id', 0)}",
            },
            "prompts":prompts
        }
        return msg


class TemporalReasoningGivenCaptionDataset(TemporalReasoningDataset):
    DATASET_ALIAS = "tr_cap"
    JSON_PATH = "meta_info/holistic_reasoning_temporal.json" 
    def _build_auxiliary_prompts(self, line: dict) -> list:
        """Overrides the hook to add a caption."""
        caption_prompt = f"Here is a caption that describes the full, uncut audio scene to help you reconstruct the original context: {line['global_caption']}\n Below are 3 audio clips:\n"
        return [{'type': 'text', 'value': caption_prompt}]
    
    def build_prompt(self, line: dict, **kwargs) -> dict:
        msg = super().build_prompt(line, **kwargs)
        msg['meta']['given_caption'] = True
        return msg




class TemporalReasoningGivenUncutAudioDataset(TemporalReasoningDataset):
    DATASET_ALIAS = "tr_uncut"
    JSON_PATH = "meta_info/holistic_reasoning_temporal.json"

    def _build_auxiliary_prompts(self, line: dict) -> list:
        """Overrides the hook to add the uncut audio."""
        return [
            {'type':'text', 'value': "An uncut audio of the entire sound scene is provided:"},
            {'type':'audio', 'value': os.path.join(self.dataset_root, line['uncut_audio_path'])},
            {'type':'text', 'value': "\n The three clips are extracted from it, please arrange them in the correct order:\n"}
        ]
    
    def build_prompt(self, line: dict, **kwargs) -> dict:
        msg = super().build_prompt(line, **kwargs)
        msg['meta']['given_uncut_audio'] = True
        return msg




class RotationBasedDataset(MCQBaseDataset):
    """Base class for datasets using option rotation (Spatial, Perception)."""
    def _prepare_options_and_audios(self, item: dict, rotate_id: int = 0, **kwargs) -> tuple:
        raw_options = item["options"]
        answer = item["answer"]
        prepared_options = raw_options[-rotate_id:] + raw_options[:-rotate_id]
        ans_idx = prepared_options.index(answer)
        answer_letter = f'<{string.ascii_uppercase[ans_idx]}>'
        return prepared_options, item["audio_paths"], answer_letter

class SpatialReasoningDataset(RotationBasedDataset):
    DATASET_ALIAS = 'sr'
    JSON_PATH="meta_info/holistic_reasoning_spatial.json"
   

class SpatialReasoningChannelwiseDataset(SpatialReasoningDataset):
    DATASET_ALIAS="sr_ch"
    JSON_PATH="meta_info/holistic_reasoning_spatial.json"

    def __init__(self, dataset_root: str):
        super().__init__(dataset_root)
        raw_saptial_audio_dir = os.path.join(self.dataset_root, "starbench_audios/holistic_reasoning/spatial")
        self.split_audio_dir = raw_saptial_audio_dir+"_split"
        if not os.path.exists(self.split_audio_dir):
            logger.info(f"Split audio directory not found. Creating it now...")
            binaural2single(raw_spatial_audio_dir, self.split_audio_dir)
            logger.info("Audio channel splitting complete.")
    
    def _build_audio_prompts(self, audio_paths: list) -> list:
        """
        [OVERRIDE] This is the ONLY method we need to change.
        It replaces the default audio prompt with the channel-wise version.
        """
        assert len(audio_paths) == 1, "Channel-wise expects a single binaural audio file per item."
        
        # Correctly get the filename and stem from the path string
        audio_filename = os.path.basename(audio_paths[0])
        audio_stem = os.path.splitext(audio_filename)[0]

        left_path = os.path.join(self.split_audio_dir, f"{audio_stem}_left.wav")
        right_path = os.path.join(self.split_audio_dir, f"{audio_stem}_right.wav")

        return [
            {'type': 'text', 'value': 'Audio 1:'},
            {'type': 'audio', 'value': left_path},
            {'type': 'text', 'value': 'Audio 2:'},
            {'type': 'audio', 'value': right_path},
            {'type': 'text', 'value': 'Audio 1 is the left-ear channel and Audio 2 is the right-ear channel.\n'}
        ]
        
    def build_prompt(self, line: int | dict, **kwargs) -> dict:
        msg = super().build_prompt(line, **kwargs)
        msg['meta']['channel_wise'] = True
        return msg
       
       


class PerceptionDataset(RotationBasedDataset):
    DATASET_ALIAS = "pc"
    JSON_PATH = "meta_info/foundation_perception.json"


    def evaluate(self, eval_file: str, robust_eval=True, reeval=False) -> dict:
        """
        Calculates final performance metrics (Total, by Category, by Sub-category)
        from a results file and returns a clean, serializable dictionary.
        """
        if not os.path.exists(eval_file) or os.path.getsize(eval_file) == 0:
            return {"error": f"Evaluation file not found or is empty: {eval_file}"}
        
        if reeval:
            tmp_path = eval_file + ".tmp"
            updated = 0
            total = 0
            # Re-evaluate each line and rewrite the source file
            with open(eval_file, "r", encoding="utf-8") as fin, \
                open(tmp_path,  "w", encoding="utf-8") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        item = json.loads(line)
                        model_resp = item.get("model_response", "")
                        # Re-evaluate this record
                        new_item = self.evaluate_item(item, model_resp)
                        fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                        updated += 1
                    except Exception as e:
                        fout.write(line + "\n")
           
            os.replace(tmp_path, eval_file)
            logger.info(f"Re-evaluated {updated}/{total} items and rewrote {eval_file}.")
           
        df = pd.read_json(eval_file, lines=True)

        # logger.info(f"Verifying the completeness of test samples in {eval_file}")
        # expected_total = len(self.data) * robustness_runs
        # assert len(df) == expected_total, \
        #     f"Row count mismatch: expected {expected_total}, got {len(df)}"
        # counts = df['id'].value_counts()
        # assert all(counts == robustness_runs), \
        #     f"Some ids do not have exactly {robustness_runs} runs"

        logger.info(f"Verifying the completeness of test samples in {eval_file}")
        id2nopts = {d['id']: len(d['options']) for d in self.data}
        expected_total = sum(id2nopts.values())
        assert len(df) == expected_total, f"Row count mismatch: expected {expected_total}, got {len(df)}"
        counts = df['id'].value_counts()
        wrong = [i for i, c in counts.items() if c != id2nopts.get(i, -1)]
        assert not wrong, f"Some ids do not have exactly expected runs: {wrong}"

        # ----------------------- By Sub-category -----------------------------
        by_subcat = {
            str(sub): self._calculate_metrics(g)   # {"AA":"..%","ACR":"..%","total_runs":..,"unique_samples":..}
            for sub, g in df.groupby('sub-category')
        }

        # Macro Average
        def pct2float(x):
            return float(x[:-1]) if isinstance(x, str) and x.endswith('%') else float(x)

        subcat_rows = []
        for sub, g in df.groupby('sub-category'):
            m = by_subcat[str(sub)]
            subcat_rows.append({
                "category": g['category'].iloc[0],
                "sub-category": sub,
                "AA": pct2float(m["AA"]),
                "ACR": pct2float(m["ACR"]),
            })
        sub_df = pd.DataFrame(subcat_rows)

        # ----------------------- By Category (macro over subcats) -----------
        by_cat = {}
        for cat, g in sub_df.groupby('category'):
            # 等权平均：每个 sub-category 的 AA/ACR 一样权重
            aa_macro  = g["AA"].mean()
            acr_macro = g["ACR"].mean()

            # 该 category 的样本统计来自原始 df 的过滤
            cat_df = df[df["category"] == cat]
            by_cat[str(cat)] = {
                "AA": f"{aa_macro:.2f}%",
                "ACR": f"{acr_macro:.2f}%",
                "n_subcategories": int(len(g)),
                "total_runs": int(len(cat_df)),
                "unique_samples": int(cat_df["id"].nunique()),
            }

        # ----------------------- Total (macro over ALL subcats) -------------
        total = {
            # 等权平均所有 sub-category
            "AA": f"{sub_df['AA'].mean():.2f}%",
            "ACR": f"{sub_df['ACR'].mean():.2f}%",
            "n_subcategories": int(len(sub_df)),
            "total_runs": int(len(df)),
            "unique_samples": int(df["id"].nunique()),
        }

        return {
            "By Sub-category": by_subcat,
            "By Category": by_cat,
            "Total": total
        }


class PerceptionSpatialDataset(PerceptionDataset):
    DATASET_ALIAS = "pc_sp"
    JSON_PATH = "meta_info/foundation_perception_spatial.json"

class PerceptionNonSpatialDataset(PerceptionDataset):
    DATASET_ALIAS = "pc_nsp"
    JSON_PATH = "meta_info/foundation_perception_nonspatial.json"


    


        



        






