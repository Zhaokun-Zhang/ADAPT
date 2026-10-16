"""Iteration loop: retrieve, allocate the per-round budget, then stop or refine."""

import json
import os

import numpy as np
import pandas as pd

from ADAPT.allocation import allocate_and_verify
from ADAPT.paths import RESULT_DIR
from ADAPT.retrieval import build_candidate_queues
from ADAPT.stopping import early_stop
from ADAPT.utils import save_json, show_time


def join_iteration(glbinfo, **args):
    glbinfo.dataset.print_info()
    while glbinfo.iter_id < args["iter_num"]:
        show_time(f"LOOP {glbinfo.iter_id}  T={args['per_round_budget']}")
        build_candidate_queues(glbinfo, **args)
        allocate_and_verify(glbinfo, **args)
        early_stop(glbinfo, **args)
        glbinfo.recorder.save()
        if glbinfo.early_stop:
            break
        glbinfo.iter_id += 1
    save_join_results(glbinfo, **args)


def save_join_results(glbinfo, **args):
    save_dir = os.path.join(
        str(RESULT_DIR), args["llm_mode"], args["mark"], glbinfo.dataset.Dname)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving join results to {save_dir}")

    joined_pairs = glbinfo.llm_joined.get_positive_pairs()
    pd.DataFrame(joined_pairs, columns=["Lid", "Rid"]).to_csv(
        os.path.join(save_dir, "joined_pairs.csv"), index=False)

    dtype = np.dtype([("", "int"), ("", "int")])
    pred = set(joined_pairs.astype(int).view(dtype=dtype).reshape(-1).tolist()) if len(joined_pairs) else set()
    actual_arr = glbinfo.dataset.joined.values.copy().astype(int)
    actual = set(actual_arr.view(dtype=dtype).reshape(-1).tolist()) if len(actual_arr) else set()
    tp, fp, fn = len(pred & actual), len(pred - actual), len(actual - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    usage = glbinfo.llm.get_usage()
    preprocess_tasks = ("data_description", "driving_or_driven", "map")

    def _cost(block):
        return {
            "total_calls": int(block.get("num_requests", 0)),
            "prompt_tokens": int(block.get("prompt_tokens", 0)),
            "completion_tokens": int(block.get("completion_tokens", 0)),
        }

    total = _cost(usage.get("total", {}))
    optimization = _cost({})
    for task in preprocess_tasks:
        if task not in usage:
            continue
        for key, val in _cost(usage[task]).items():
            optimization[key] += val
    execution = {key: total[key] - optimization[key] for key in total}
    save_json({
        "metrics": {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": round(precision, 6),
            "Recall": round(recall, 6),
            "F1": round(f1, 6),
        },
        "optimization_cost": {glbinfo.llm.model: optimization},
        "execution_cost": {glbinfo.llm.model: execution},
        "total_cost": {glbinfo.llm.model: total},
    }, os.path.join(save_dir, "metrics.json"), indent=2)
    with open(os.path.join(save_dir, "config.json"), "w") as handle:
        json.dump(args, handle, indent=2, default=str)
