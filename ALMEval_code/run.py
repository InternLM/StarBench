import argparse
import os
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
import datetime
import shutil
from loguru import logger
import sys
import pandas as pd
import glob
from datasets import build_dataset
from models import build_model
from pathlib import Path
import time
from typing import Dict, Any, Set, List, Tuple
import traceback
import subprocess

# GET the number of GPUs on the node without importing libs like torch
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        return []


RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE",1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
# cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
master_addr = os.environ.get('MASTER_ADDR', '<unset>')
master_port = os.environ.get('MASTER_PORT', '<unset>')


GPU_LIST = get_gpu_list()
if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
    NGPU = len(GPU_LIST)
    assert NGPU >= LOCAL_WORLD_SIZE, "The number of processes should be less than or equal to the number of GPUs"
    GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
    DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
    CUDA_VISIBLE_DEVICES = [str(i) for i in GPU_LIST[DEVICE_START_IDX: DEVICE_START_IDX + GPU_PER_PROC]]
    CUDA_VISIBLE_DEVICES = ','.join(CUDA_VISIBLE_DEVICES)
    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    print(
        f'RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE}, '
        f'LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}, '
        f'MASTER_ADDR={master_addr} MASTER_PORT={master_port}',
        flush=True
    )


def setup_logging(rank, log_dir):
    """Sets up file and console logging for each process."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'rank_{rank}.log')
    logger.remove()
    logger.add(log_file, format="{time:HH:mm:ss} | {message}", level="INFO", rotation="10 MB")
    if rank == 0:
        logger.add(sys.stdout, format="{time:HH:mm:ss} | {message}", level="INFO")
    logger.info(f"[logger-ready] rank={rank} pid={os.getpid()}")



def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of objects; skip bad lines."""
    if not os.path.exists(path):
        return []
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                data.append(json.loads(s))
            except json.JSONDecodeError:
                print(f"[warn] Skip bad line {ln} in {path}")
    return data

def read_unique_ids(path: str, key: str = "unique_id") -> Set[str]:
    """Collect unique IDs from a JSONL file."""
    return {obj[key] for obj in read_jsonl(path) if key in obj}

def write_jsonl_append(path: str, records: List[Dict[str, Any]], do_fsync: bool = True) -> int:
    """Append all records to a JSONL file; return count."""
    if not records:
        return 0
    lines = '\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n'
    with open(path, 'a', encoding='utf-8') as f:
        f.write(lines)
        f.flush()
        if do_fsync:
            os.fsync(f.fileno())
    return len(records)


def merge_one_dataset(
    temp_file_pattern: str,
    final_file: str,
    unique_key: str = "unique_id",
    remove_sources: bool = True,
) -> Tuple[int, int]:
    """
    Merge all tmp JSONL files (matched by pattern) into final_file (append mode).
    - Strictly checks duplicates by `unique_key` across tmp files and with final_file.
    - Raises ValueError on any conflict.
    - Optionally removes tmp files after a successful merge.
    Returns: (num_tmp_files, num_appended_records)
    """
    temp_files = sorted(glob.glob(temp_file_pattern))
    if not temp_files:
        print("[merge_one_dataset] No tmp files to merge.")
        return 0, 0

    Path(final_file).parent.mkdir(parents=True, exist_ok=True)
    Path(final_file).touch(exist_ok=True)

    # IDs already present in the final file
    final_ids: Set[str] = read_unique_ids(final_file, key=unique_key)

    seen: Set[str] = set()
    owners: Dict[str, str] = {}
    collected: List[Dict[str, Any]] = []

    # Load all tmp files and detect conflicts
    for tf in temp_files:
        for obj in read_jsonl(tf):
            uid = obj.get(unique_key)
            if uid is None:
                continue
            # Conflict: already in final file
            if uid in final_ids:
                raise ValueError(
                    f"\n[merge_one_dataset] Conflict detected!\n"
                    f"  Unique ID   : {uid}\n"
                    f"  Source file : {tf}\n"
                    f"  Problem     : This ID already exists in the final file: {final_file}\n"
                    f"  Action      : Please manually merge these tmp results into the final JSONL file, "
                    f"then re-run the program.\n"
                )

            # Conflict: duplicate among tmp files
            if uid in seen:
                raise ValueError(
                    f"\n[merge_one_dataset] Conflict detected between tmp files!\n"
                    f"  Unique ID   : {uid}\n"
                    f"  First file  : {owners[uid]}\n"
                    f"  Duplicate in: {tf}\n"
                    f"  Problem     : The same unique_id appears in multiple tmp files.\n"
                    f"  Action      : Please manually merge these tmp files into the final JSONL file, "
                    f"then re-run the program.\n"
                )
            seen.add(uid)
            owners[uid] = tf
            collected.append(obj)

    if not collected:
        print("[merge_one_dataset] Nothing to append.")
        # Optionally clean up empty tmp files
        if remove_sources:
            for tf in temp_files:
                try:
                    os.remove(tf)
                except OSError as e:
                    print(f"[merge_one_dataset] Failed to remove {tf}: {e}")
        return len(temp_files), 0

    # Append to final (durable via fsync)
    appended = write_jsonl_append(final_file, collected, do_fsync=True)

    # Remove sources if requested
    if remove_sources:
        for tf in temp_files:
            try:
                os.remove(tf)
            except OSError as e:
                print(f"[merge_one_dataset] Failed to remove {tf}: {e}")

    print(f"[merge_one_dataset] Merged {len(temp_files)} files, appended {appended} records -> {final_file}")
    return len(temp_files), appended


def process_dataset(args, dataset, model):
    """
    Handles the complete inference and evaluation pipeline for a single dataset. Assign different subsets to each process
    """
    # --- Setup Paths ---
    model_data_dir = os.path.join(args.work_dir, args.model, dataset.DATASET_ALIAS)
    final_results_file = os.path.join(model_data_dir, 'infer_results.jsonl')
    eval_file = os.path.join(model_data_dir, 'performance.json')
    temp_results_file = os.path.join(model_data_dir, f'tmp_results_rank_{RANK}.jsonl')
    tmp_glob_pattern  = os.path.join(model_data_dir, 'tmp_results_rank_*.jsonl')
    done_flag_pattern = os.path.join(model_data_dir, f'done_flag_{WORLD_SIZE}_*.done') 
    done_flag_file = os.path.join(model_data_dir, f'done_flag_{WORLD_SIZE}_{RANK}.done') 


    if not args.force_reinfer and os.path.exists(eval_file):
        if RANK == 0:
            with open(eval_file, 'r', encoding='utf-8') as f:
                performance = json.load(f)
            print(f"Using existing evaluation results for {dataset.DATASET_ALIAS}, skipping inference (loaded from {eval_file}).")     
            return performance
        else:
            # Non-zero ranks just exit early; rank 0 will return the previous result.
            return None

    #---- preprocessing before new run ---
    if RANK == 0:
        if args.force_reinfer and os.path.exists(model_data_dir):
            print(f"--force-reinfer: removing directory {model_data_dir}")
            shutil.rmtree(model_data_dir)
        
        os.makedirs(model_data_dir, exist_ok=True)

        for f in glob.glob(done_flag_pattern):
            try:
                os.remove(f)
            except OSError:
                pass

        leftover = sorted(glob.glob(tmp_glob_pattern))
        if leftover:
            print("Found leftover tmp files. Merging before new run...")
            merge_one_dataset(tmp_glob_pattern, final_results_file)
            print("Leftover tmp merged.")
    
    if WORLD_SIZE > 1:
        dist.barrier()

    setup_logging(RANK, os.path.join(model_data_dir, 'logs'))
        
    # --- Resume completed IDs (rank0 loads, then broadcast) ---
    completed_ids = set()  # define for all ranks

    if WORLD_SIZE > 1:
        dist.barrier() 

    if not args.force_reinfer and os.path.exists(final_results_file):
        completed_ids = read_unique_ids(final_results_file, key="unique_id")

    logger.info(
        f"Starting inference from scratch for {dataset.DATASET_ALIAS} due to --force-reinfer."
        if args.force_reinfer else
        f"Resuming inference. Found {len(completed_ids)} completed runs for {dataset.DATASET_ALIAS}."
    )
    if WORLD_SIZE > 1:
        dist.barrier()
    
    if args.debug:
        dataset.set_demo_mode() 
        if RANK == 0: 
            logger.info("Debug mode enabled — processing a limited number of samples.")

    # --- Distribute Data ---
    items_this_rank = dataset.data[RANK::WORLD_SIZE]
    
    # --- Robust Evaluation Main Loop ---
    Path(temp_results_file).touch(exist_ok=True)
    pbar = tqdm(total=len(items_this_rank), desc=f"Rank {RANK} World Size:{WORLD_SIZE}| {dataset.DATASET_ALIAS}", disable=(RANK != 0))
    dump_batch = []
    dump_batch_size = 10
    for item in items_this_rank:
        # Setup robust evaluation runs
        if args.robust_eval:
            if dataset.DATASET_ALIAS.startswith("pc"):  
                params_list = [{'rotate_id': i} for i in range(len(item['options']))] # Perception tasks: runs = number of options
            else:
                params_list = [{'rotate_id': i} for i in range(3)] # Reasoning tasks: fixed runs
        else:
            params_list = [{'rotate_id': 0}] # Single evaluation
        
        for params in params_list:
            unique_id = f"{item['id']}@{params.get('rotate_id', 0)}"
            # print('unique_id:', unique_id,  unique_id in completed_ids)
            if unique_id in completed_ids:
                continue

            msg = dataset.build_prompt(item, **params)
            try:
                response = model(msg) 
            except Exception as e:
                logger.error(f"Model generation failed for item {item['id']} with params {params}. Error: {e}")
                response = f"ERROR: {e}" 

            if args.debug and RANK == 0:
                print('msg:', msg)
                print(f'{args.model} response:', response)
                print('-'*50)

            result_info = dataset.evaluate_item(msg['meta'], response)
            result_info['unique_id'] = unique_id
            dump_batch.append(result_info)
            
            if len(dump_batch) >= dump_batch_size:
                write_jsonl_append(temp_results_file, dump_batch, do_fsync=True)
                dump_batch.clear()   

        pbar.update(1)

    if dump_batch:
        write_jsonl_append(temp_results_file, dump_batch, do_fsync=True)
        dump_batch.clear()     
    pbar.close()

    # Write a file to indicate this rank is done
    with open(done_flag_file, 'w') as f:
        f.write('done')


    #  --- Synchronization, Merging, and Final Evaluation ---
    if WORLD_SIZE > 1:
        dist.barrier()

    # Rank 0 needs to wait for other ranks to finish, then merge results and evaluate
    if RANK == 0:
        timeout_s, waited = 100, 0
        logger.info("waiting for all ranks to finish")
        while True:
            done_files = sorted(glob.glob(done_flag_pattern))
            print(f'Found {len(done_files)} done_files')
            if len(done_files) == WORLD_SIZE:
                for done_file in done_files:
                        os.remove(done_file)
                break
            if waited >= timeout_s:
                logger.warning(f"Timeout waiting tmp files more than {timeout_s}s. Found {len(done_files)}.")
                break
            time.sleep(5); waited +=5
            logger.info(f'waiting for other ranks to finish, time elapsed: {waited}s')
                
        logger.info(f"Inference complete for all ranks. Merging results...")
        merge_one_dataset(tmp_glob_pattern, final_results_file)
        logger.info(f"Merging complete. Evaluating all results...")
        performance = dataset.evaluate(final_results_file, robust_eval=args.robust_eval)
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(performance, f, indent=4, ensure_ascii=False)
        logger.info(f"Evaluation complete. Performance report saved to {eval_file}")
        return performance

    return None


def summarize_and_save(all_performances: dict, work_dir: str, model: str, fname: str = "performance_summary.csv"):
    """
    Summarize AA/ACR across datasets, append a 'Final MA' row, print, and save to CSV.
    all_performances: {dataset_alias: {"AA": "...", "ACR": "...", "unique_samples": int, ...}, ...}
    """
    if not all_performances:
        logger.warning("No performance results were generated to summarize.")
        return None

    summary_data = [
        {
            'Dataset': alias,
            'AA': perf.get('AA', 'N/A'),
            'ACR': perf.get('ACR', 'N/A'),
            'Samples': perf.get('unique_samples', 'N/A')
        }
        for alias, perf in all_performances.items()
    ]
    df = pd.DataFrame(summary_data)

    # Convert AA/ACR to numeric safely (works whether they're "xx.xx%" or plain numbers)
    def _to_num(series):
        s = series.astype(str).str.replace('%', '', regex=False)
        return pd.to_numeric(s, errors='coerce')

    df['AA']  = _to_num(df['AA'])
    df['ACR'] = _to_num(df['ACR'])
    df['Samples'] = pd.to_numeric(df['Samples'], errors='coerce')

    final_row = {
        'Dataset': 'Final MA',
        'AA': df['AA'].mean(skipna=True),
        'ACR': df['ACR'].mean(skipna=True),
        'Samples': df['Samples'].sum(skipna=True) 
    }
    summary_df = pd.concat([df, pd.DataFrame([final_row])], ignore_index=True)

    print("\n" + "-" * 80)
    try:
        print(summary_df.to_markdown(index=False, floatfmt=".3f") + "\n")
    except Exception:
        print(summary_df.to_string(index=False) + "\n")

    # save
    out_dir = os.path.join(work_dir, model)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, fname)
    summary_df.to_csv(save_path, index=False)
    logger.info(f"Saved summary to: {save_path}")
    return summary_df

def do_reeval(args):
    """Re-run evaluation on existing inference results."""
    logger.info(f"Re-evaluation mode enabled. Model: {args.model}")
    datasets = build_dataset(args.data, dataset_root=args.dataset_root)
    all_performances = {}
    all_alias_str ="+".join([dataset.DATASET_ALIAS for dataset in datasets])
    for dataset in datasets:
        logger.info("=" * 80)
        logger.info(f"Re-evaluating dataset: {dataset.DATASET_ALIAS}")
        if args.debug:
           dataset.set_demo_mode() 
           logger.info("Debug mode enabled — processing a limited number of samples.")
        final_results_file = os.path.join(
            args.work_dir, args.model, dataset.DATASET_ALIAS, 'infer_results.jsonl'
        )
        assert os.path.exists(final_results_file), f"File not found: {final_results_file}"
        
        performance = dataset.evaluate(final_results_file, robust_eval=args.robust_eval, reeval=True)
        eval_file = os.path.join(
            args.work_dir, args.model, dataset.DATASET_ALIAS, 'performance.json'
        )
        os.makedirs(os.path.dirname(eval_file), exist_ok=True)
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(performance, f, indent=4, ensure_ascii=False)
        all_performances[dataset.DATASET_ALIAS] = performance.get("Total", {})

    logger.info("=" * 100)
    logger.info(f"Final Summary (Re-eval) for Model: {args.model}")
    logger.info("=" * 100)
    summarize_and_save(all_performances, args.work_dir, args.model, fname=f"reeval_summary_{all_alias_str}.csv")


def main(args):
    # init_distributed(args)
    if WORLD_SIZE > 1:
        import torch.distributed as dist
        # torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    if RANK == 0: 
        logger.info(f"Loading model: {args.model}")
    model = None if args.reeval else build_model(args.model)
    
    if RANK == 0: 
        logger.info(f"Building datasets for: {args.data}")
    datasets = build_dataset(args.data, dataset_root=args.dataset_root)

    all_performances = {}
    all_alias_str ="+".join([dataset.DATASET_ALIAS for dataset in datasets])
    for dataset in datasets:
        if WORLD_SIZE > 1:
            dist.barrier()
        if RANK == 0:
            logger.info("=" * 80)
            logger.info(f"Processing dataset: {dataset.DATASET_ALIAS}")

        performance = process_dataset(args, dataset, model)

        if WORLD_SIZE > 1:
            dist.barrier()
        if RANK == 0 and performance:
            all_performances[dataset.DATASET_ALIAS] = performance.get("Total", {})

    # Summarize the performance on all datasets
    if RANK == 0:
        logger.info("=" * 100)
        logger.info(f"Final Summary for Model: {args.model}")
        logger.info("=" * 100)
        summarize_and_save(all_performances, args.work_dir, args.model, fname=f"performance_summary_{all_alias_str}.csv")
        

    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evaluation for audio-language models.")
    parser.add_argument('--data', type=str, nargs='+', required=True, help='List of dataset aliases or groups (e.g., sr tr_all)')
    parser.add_argument('--model', type=str, required=True, help="The alias of the model to evaluate, as defined in 'models/models.yaml' (e.g., 'qwen25-omni', 'kimi').")
    parser.add_argument('--dataset_root', type=str, default='../STAR-Bench', help='Root directory of datasets')
    parser.add_argument('--work-dir', type=str, default='./eval_results', help='Working directory to save results')
    parser.add_argument('--robust-eval', type=bool, default=True, help='Whether to perform robustness evaluation with multiple runs')
    parser.add_argument('--reeval', action='store_true', help='Re-run evaluation on existing results with latest logic')
    parser.add_argument('--force-reinfer', action='store_true', help='Force re-inference for all samples. If not set, the program will automatically resume from previous inference outputs.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with only a few samples')
    args = parser.parse_args()

    try:
        if args.reeval:
            do_reeval(args)
        else:
            main(args)
    except Exception as e:
        logger.exception(f"FATAL exception on process (env RANK={os.getenv('RANK')}): {e}")
        traceback.print_exc()
        raise
    finally: 
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as _:
            pass
