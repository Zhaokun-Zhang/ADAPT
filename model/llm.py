"""LLM client for profiling, feature extraction, and pair verification.

Modes
-----
normal
    Every call goes to the model.
cache
    Pair-verification labels are reused from ``cache/llm_match/<dataset>.pkl``.
    A miss calls the model and is written back. Profiling and map extraction
    always call the model.
simulation
    Pair verification is answered from the dataset ground truth. Profiling and
    map extraction still call the model.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import litellm
import numpy as np

from ADAPT.paths import CACHE_DIR, ROOT
from ADAPT.sparse_label_matrix import SparseLabelMatrix
from ADAPT.utils import load_pickle, save_pickle


def basic_parser(json_str: str):
    json_str = json_str.replace("```json", "").replace("`", "")
    try:
        return json.loads(json_str)
    except Exception:
        pass
    try:
        return eval(json_str)
    except Exception:
        pass
    return None


def single_matching_task_parser(llm_call_info: dict, llm_result: dict):
    content = llm_result["content"]
    if content is None:
        return None
    result = basic_parser(content)
    if not isinstance(result, dict):
        return None
    norm = {str(k).lower(): v for k, v in result.items()}
    verdict = norm.get("verdict")
    if verdict is None:
        return None
    if isinstance(verdict, str):
        verdict = verdict.strip().lower() in ("true", "yes", "1")
    return {"verdict": bool(verdict)}


def data_description_task_parser(llm_call_info: dict, llm_result: dict):
    content = llm_result["content"]
    if content is None:
        return None
    result = basic_parser(content)
    if not isinstance(result, dict):
        return None
    norm = {str(k).lower(): v for k, v in result.items()}
    left, right = norm.get("l_description"), norm.get("r_description")
    if left is None or right is None:
        return None
    return {"L_description": str(left), "R_description": str(right)}


def driving_or_driven_task_parser(llm_call_info: dict, llm_result: dict):
    content = llm_result["content"]
    if content is None:
        return None
    result = basic_parser(content)
    if not isinstance(result, dict):
        return None
    norm = {str(k).lower(): v for k, v in result.items()}
    driving = norm.get("driving_table")
    if driving is None:
        return None
    driving = str(driving).strip().upper()
    if driving not in ("L", "R"):
        if "L" in driving and "R" not in driving:
            driving = "L"
        elif "R" in driving and "L" not in driving:
            driving = "R"
        else:
            return None
    return {
        "driving_table": driving,
        "rule_applied": norm.get("rule_applied"),
        "reason": norm.get("reason"),
    }


def mapping_task_parser(llm_call_info: dict, llm_result: dict):
    content = llm_result["content"]
    if content is None:
        return None
    result = basic_parser(content)
    if result is not None:
        return result
    content = content.replace('"', "").replace("'", "").replace("[", "").replace("]", "").replace("`", "")
    return [feat.strip() for feat in content.split(",") if feat.strip()]


def load_env(env_file: Path = None):
    if env_file is None:
        for candidate in (ROOT / ".env", Path.cwd() / ".env"):
            if candidate.exists():
                env_file = candidate
                break
        else:
            env_file = ROOT / ".env"
    env: Dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    missing = [key for key in ("MODEL", "API_BASE", "API_KEY") if key not in env]
    if missing:
        raise RuntimeError(
            f"Missing {missing} in {env_file}. "
            "Create Adapt-code/.env with MODEL, API_BASE, and API_KEY."
        )
    return env["MODEL"], env["API_BASE"], env["API_KEY"]


_PARSERS = {
    "map": mapping_task_parser,
    "match_single": single_matching_task_parser,
    "data_description": data_description_task_parser,
    "driving_or_driven": driving_or_driven_task_parser,
}


class LLM:
    def __init__(self, max_workers: int = 16, mode: str = "cache"):
        self.cache_dir = CACHE_DIR / "llm_match"
        self.model, self.api_base, self.api_key = load_env()
        self.max_workers = max_workers
        self.mode = mode
        self.joined = None
        self.cache_path = None
        self.llm_joined = None
        self.init_usage_dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "num_requests": 0,
        }
        self.usage_dict = {
            "total": self.init_usage_dict.copy(),
            "current_batch": self.init_usage_dict.copy(),
        }

    def build_llm_joined_cache(self, data):
        if self.mode != "cache":
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = str(self.cache_dir / f"{data.Dname}.pkl")
        if not os.path.exists(self.cache_path):
            llm_joined = SparseLabelMatrix(len(data.L), len(data.R))
            save_pickle(llm_joined.to_dict(), self.cache_path)

    def build_simulation_joined(self, joined_pairs: np.ndarray):
        joined = defaultdict(dict)
        for lid, rid in joined_pairs:
            joined[int(lid)][int(rid)] = True
        self.joined = joined

    def __call__(self, llm_call_info_list: List[Dict[str, Any]], enable_thinking: bool = False, **args):
        if not llm_call_info_list:
            return []
        tasks = {item["Task"] for item in llm_call_info_list}
        if tasks == {"match_single"}:
            if self.mode == "simulation":
                return self._simulation_mode_single_llm_call(llm_call_info_list)
            if self.mode == "cache":
                return self._cache_mode_single_llm_call(llm_call_info_list, enable_thinking, **args)
        return self._normal_mode_llm_call(llm_call_info_list, enable_thinking, **args)

    def _normal_mode_llm_call(self, llm_call_info_list, enable_thinking, prt=False, **args):
        prompts = [item["Prompt"] for item in llm_call_info_list]
        tasks = [item["Task"] for item in llm_call_info_list]
        if prt:
            print(f"LLM calling tasks={tasks}, enable_thinking={enable_thinking}")
        llm_results = self._batch_completion(
            batch_messages=prompts, enable_thinking=enable_thinking, tasks=tasks, **args)
        parsed = []
        for call_info, task, result in zip(llm_call_info_list, tasks, llm_results):
            parser = _PARSERS.get(task)
            parsed.append(parser(call_info, result) if parser else result["content"])
        return parsed

    def _simulation_mode_single_llm_call(self, llm_call_info_list, **args):
        joined = self.joined or {}
        results = []
        for info in llm_call_info_list:
            lid, rid = info["Lid"], info["Rid"]
            results.append({"verdict": bool(joined.get(lid, {}).get(rid, False))})
        self._num_requests_simulation(llm_call_info_list)
        return results

    def _load_match_cache(self) -> SparseLabelMatrix:
        if self.llm_joined is not None:
            return self.llm_joined
        cached = load_pickle(self.cache_path)
        if isinstance(cached, np.ndarray):
            matrix = SparseLabelMatrix.from_dense(cached)
        elif isinstance(cached, dict):
            matrix = SparseLabelMatrix.from_dict(cached)
        else:
            raise ValueError(f"Unknown LLM cache format: {type(cached)}")
        self.llm_joined = matrix
        return matrix

    def _cache_mode_single_llm_call(self, llm_call_info_list, enable_thinking, **args):
        matrix = self._load_match_cache()
        results: List = [None] * len(llm_call_info_list)
        miss_idx, miss_list = [], []
        for i, info in enumerate(llm_call_info_list):
            cached = matrix[info["Lid"], info["Rid"]]
            if cached == 1:
                results[i] = {"verdict": True}
            elif cached == 0:
                results[i] = {"verdict": False}
            else:
                miss_idx.append(i)
                miss_list.append(info)
        if miss_list:
            miss_results = self._normal_mode_llm_call(
                miss_list, enable_thinking, update_usage=False, **args)
            for i, res in zip(miss_idx, miss_results):
                results[i] = res
                if res is None:
                    continue
                lid = llm_call_info_list[i]["Lid"]
                rid = llm_call_info_list[i]["Rid"]
                matrix[lid, rid] = 1 if res.get("verdict") else 0
            save_pickle(matrix.to_dict(), self.cache_path)
        self._input_tokens_simulation(llm_call_info_list)
        return results

    def _input_tokens_simulation(self, llm_call_info_list):
        self._init_current_usage()
        for item in llm_call_info_list:
            prompt_tokens = litellm.token_counter(model=self.model, messages=item["Prompt"])
            self._update_usage(
                {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}, item["Task"])

    def _num_requests_simulation(self, llm_call_info_list):
        self._init_current_usage()
        for item in llm_call_info_list:
            self._update_usage({}, item["Task"])

    def _init_current_usage(self):
        self.usage_dict["current_batch"] = self.init_usage_dict.copy()

    def _update_usage(self, usage: Dict[str, int], task: str = None):
        if usage is None:
            return

        def usage_add(dic: dict, new_usage):
            for key in dic.keys():
                dic[key] += new_usage.get(key, 0)
            dic["num_requests"] += 1

        usage_add(self.usage_dict["total"], usage)
        usage_add(self.usage_dict["current_batch"], usage)
        if task is not None:
            self.usage_dict.setdefault(task, self.init_usage_dict.copy())
            usage_add(self.usage_dict[task], usage)

    def _batch_completion(self, batch_messages, enable_thinking: bool = False, tasks=None, **args):
        if tasks is None:
            tasks = [None] * len(batch_messages)
        self._init_current_usage()
        responses = litellm.batch_completion(
            model=self.model,
            messages=batch_messages,
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=0,
            max_workers=min(self.max_workers, len(batch_messages)),
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        results = []
        for resp, task in zip(responses, tasks):
            if isinstance(resp, Exception):
                results.append({
                    "content": None, "error": str(resp), "usage": None,
                    "task": task, "model": self.model,
                })
                continue
            if args.get("update_usage", True):
                self._update_usage(resp.usage, task=task)
            results.append({
                "content": resp.choices[0].message.content,
                "usage": resp.usage,
                "task": task,
                "model": self.model,
                "reasoning": getattr(resp.choices[0].message, "reasoning_content", None)
                if enable_thinking else None,
            })
        return results

    def prt_usage(self):
        for task, usage in self.usage_dict.items():
            print(f"{task} llm usage:")
            for key, val in usage.items():
                print(f"  {key}: {val:,}")

    def get_usage(self) -> Dict[str, Dict[str, int]]:
        return self.usage_dict
