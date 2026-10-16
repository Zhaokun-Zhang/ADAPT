"""Sliding-window UCB allocation and single-pair LLM verification."""

import math
from collections import deque
from copy import deepcopy
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np

from ADAPT.paths import PROMPT_DIR
from ADAPT.utils import load_json, show_time
from model import LLM


def ucb_allocation(
    candidate_list: List[List[Tuple[Any, bool]]],
    budget: int,
    window: int,
    reward_func: Callable[[Any], int],
    is_labeled_func: Callable[[Any], bool],
    alpha: float = 2.0,
    diagnostics: Dict[str, Any] = None,
):
    """Per-channel sliding-window UCB.

    After t fresh verifications, an arm's score is
    mean + sqrt(alpha * log(min(t, window)) / n), using that arm's latest
    `window` rewards. Already labeled pairs are skipped and do not spend budget.
    """
    n_arms = len(candidate_list)
    if n_arms == 0 or budget <= 0:
        return [0] * n_arms, 0, set()
    if window < 1:
        raise ValueError("window must be a positive integer")

    pos = [0] * n_arms
    win = [deque() for _ in range(n_arms)]
    win_sum = [0] * n_arms
    total_reward = 0
    selected: Set[Any] = set()
    fresh_verified = 0
    skipped = 0
    fresh_by_arm = [0] * n_arms
    positive_by_arm = [0] * n_arms
    t = 0

    while fresh_verified < budget:
        best_i = next(
            (i for i in range(n_arms) if fresh_by_arm[i] == 0 and pos[i] < len(candidate_list[i])),
            -1,
        )
        if best_i == -1:
            best_score = -math.inf
            for i in range(n_arms):
                if pos[i] >= len(candidate_list[i]):
                    continue
                n_obs = len(win[i])
                if n_obs == 0:
                    score = math.inf
                else:
                    mean = win_sum[i] / n_obs
                    bonus = math.sqrt(alpha * math.log(max(min(t, window), 1)) / n_obs)
                    score = mean + bonus
                if score > best_score:
                    best_score, best_i = score, i
        if best_i == -1:
            break

        element = candidate_list[best_i][pos[best_i]][0]
        pos[best_i] += 1
        if is_labeled_func(element):
            selected.add(element)
            skipped += 1
            continue
        reward = int(reward_func(element))
        win[best_i].append(reward)
        win_sum[best_i] += reward
        if len(win[best_i]) > window:
            win_sum[best_i] -= win[best_i].popleft()
        total_reward += reward
        fresh_by_arm[best_i] += 1
        positive_by_arm[best_i] += reward
        selected.add(element)
        fresh_verified += 1
        t += 1

    if fresh_verified < budget:
        print(f"[ucb] candidate pool exhausted: fresh_verified={fresh_verified} < T={budget} "
              f"(skipped={skipped}).")
    print(f"[ucb] fresh_verified={fresh_verified} skipped={skipped} depth={pos}")
    if diagnostics is not None:
        diagnostics.update({
            "budget_spent": int(fresh_verified),
            "fresh_unique": int(fresh_verified),
            "skipped": int(skipped),
            "positions": pos[:],
            "fresh_by_arm": fresh_by_arm,
            "positive_by_arm": positive_by_arm,
        })
    return pos[:], total_reward, selected


def _single_match_prompt(data, left_text: str, right_text: str):
    prompt = deepcopy(load_json(str(PROMPT_DIR / "single_matching.template")))
    for message in prompt:
        if isinstance(message["content"], list):
            message["content"] = "".join(message["content"])
    prompt[0]["content"] = prompt[0]["content"].replace(
        "#OUTPUT_FORMAT#", '{"verdict": true/false}')
    prompt[1]["content"] = (prompt[1]["content"]
                            .replace("#PREDICATE#", data.predicate)
                            .replace("#LEFT#", left_text)
                            .replace("#RIGHT#", right_text)
                            .replace("#CONTEXT_BLOCK#", ""))
    return prompt


def verify_pair(glbinfo, lid: int, rid: int) -> int:
    """Ask the LLM whether (lid, rid) satisfies the predicate. Returns 0 or 1."""
    lid, rid = int(lid), int(rid)
    data = glbinfo.dataset
    llm: LLM = glbinfo.llm
    call = {
        "Task": "match_single",
        "Prompt": _single_match_prompt(data, str(data.L.values[lid, 1]), str(data.R.values[rid, 1])),
        "Lid": lid,
        "Rid": rid,
    }
    result = None
    for _ in range(3):
        result = llm([call], enable_thinking=False)[0]
        if result is not None and "verdict" in result:
            break
    if result is None or "verdict" not in result:
        raise RuntimeError(f"Pair verification failed for ({lid}, {rid}).")
    label = 1 if result["verdict"] else 0
    glbinfo.llm_joined[lid, rid] = label
    return label


def _log_pull(glbinfo, lid: int, rid: int, reward: int):
    pair = (int(lid), int(rid))
    if pair in glbinfo.pull_logged_pairs:
        return
    glbinfo.pull_logged_pairs.add(pair)
    glbinfo.pull_log.append((pair[0], pair[1], int(reward), int(glbinfo.iter_id)))


def allocate_and_verify(glbinfo, **args):
    show_time("Matching start")
    pos_cand = glbinfo.pos_cand
    candidate_list = []
    for feature_type in pos_cand.feature_type_list:
        arm = []
        for lid, rid, _fid in getattr(pos_cand, f"{feature_type}_list"):
            arm.append(((int(lid), int(rid)), False))
        candidate_list.append(arm)

    def reward_func(pair):
        reward = verify_pair(glbinfo, pair[0], pair[1])
        _log_pull(glbinfo, pair[0], pair[1], reward)
        return reward

    def is_labeled(pair):
        return glbinfo.llm_joined[int(pair[0]), int(pair[1])] >= 0

    window = int(1000 * np.log(glbinfo.iter_id + np.e))
    diagnostics = {}
    allocation, _reward, _selected = ucb_allocation(
        candidate_list, int(glbinfo.total_quota), window=window,
        reward_func=reward_func, is_labeled_func=is_labeled, diagnostics=diagnostics,
    )
    recorder = glbinfo.recorder
    recorder.add("alloc_budget_spent", diagnostics["budget_spent"])
    recorder.add("alloc_skipped", diagnostics["skipped"])
    for arm, feature_type in enumerate(pos_cand.feature_type_list):
        recorder.add(f"alloc_{feature_type}_depth", diagnostics["positions"][arm])
        recorder.add(f"alloc_{feature_type}_fresh", diagnostics["fresh_by_arm"][arm])
        recorder.add(f"alloc_{feature_type}_TP", diagnostics["positive_by_arm"][arm])
        feat_list = getattr(pos_cand, f"{feature_type}_list")
        setattr(pos_cand, f"{feature_type}_list", feat_list[:allocation[arm]])

    pos_cand.trans_list2cand()
    pos_cand.union_all_feature_cand()
    glbinfo.all_pos_cand_list.append(pos_cand)
    glbinfo.all_pos_cand.union_other(pos_cand)
    glbinfo.evaluate()
    return allocation
