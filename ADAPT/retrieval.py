"""Per-channel retrieval against the fixed driven-side index."""

from collections import defaultdict
from typing import Dict, List

import numpy as np

from ADAPT.candidate import CandidateSet
from ADAPT.paths import RETRIEVAL_OVER_RATIO
from ADAPT.refinement import dl_finetune
from ADAPT.sorting import sorting_main
from ADAPT.utils import show_time


class Retrieval:
    def __init__(self, glbinfo, feature_type: str, **args):
        self.args = args
        self.glbinfo = glbinfo
        self.iter_id = glbinfo.iter_id
        self.feature_type = feature_type
        self.feat_dict = glbinfo.dataset.features[feature_type]
        self.llm_joined = glbinfo.llm_joined
        self.Lid2fids: Dict[int, List[int]] = self.feat_dict["docid2fids"]
        self.feat_emb = self.feat_dict["feat_emb"]
        self.Remb = glbinfo.dataset.Remb
        self.Rindex = glbinfo.Rindex
        self.X = self.feat_emb
        if glbinfo.iter_id != 0:
            self.finetune()

    def finetune(self):
        self.X, self.dl_info = dl_finetune(self, **self.args)

    def retrieval_one_quota(self, lid: int, quota: int):
        fids = self.Lid2fids.get(lid, [])
        if len(fids) == 0 or quota <= 0:
            return {}
        fids = np.asarray(fids)
        n_right = self.llm_joined.shape[1]
        max_topk = max(1, min(int(getattr(self.Rindex, "ntotal", n_right)), n_right))
        per_feature = int(np.ceil(quota / len(fids)))
        topk = max(1, min(int(per_feature * (self.iter_id + 1) * 2), max_topk))
        scores, ids = self.Rindex.search(np.ascontiguousarray(self.X[fids], dtype=np.float32), topk)

        cand = defaultdict(list)
        seen = set()
        left = quota
        depth = 0
        while left > 0:
            if depth >= topk:
                if topk >= max_topk:
                    break
                topk = min(topk * 2, max_topk)
                scores, ids = self.Rindex.search(
                    np.ascontiguousarray(self.X[fids], dtype=np.float32), topk)
                continue
            added = 0
            for i, fid in enumerate(fids):
                rid = int(ids[i, depth])
                if rid < 0 or rid >= n_right or rid in seen:
                    continue
                if self.llm_joined[lid, rid] >= 0:
                    continue
                cand[int(fid)].append([rid, float(scores[i, depth])])
                seen.add(rid)
                left -= 1
                added += 1
                if left <= 0:
                    return dict(cand)
            depth += 1
            if added == 0 and depth >= topk and topk >= max_topk:
                break
        return dict(cand)

    def equal_quota_retrieval(self, quota: int):
        candidate_dict = defaultdict(list)
        candidate_list = []
        if quota <= 0 or not self.Lid2fids:
            return candidate_dict, candidate_list
        per_record = int(np.ceil(quota / len(self.Lid2fids)))
        for lid in self.Lid2fids:
            for fid, pairs in self.retrieval_one_quota(lid, per_record).items():
                for rid, score in pairs:
                    candidate_dict[(lid, rid)].append(fid)
                    candidate_list.append((lid, rid, fid, score))
        if len(candidate_dict) < quota:
            print(f"Warning: {self.feature_type} retrieved {len(candidate_dict)} "
                  f"unique pairs, fewer than {quota}.")
        return candidate_dict, candidate_list


def retrieval_main(glbinfo, feature_type: str, total_quota: int, **args):
    print(f"Feature: 【 {feature_type} 】")
    retrieval = Retrieval(glbinfo, feature_type, **args)
    glbinfo.dataset.features[feature_type]["retrieval"] = retrieval
    retrieval_quota = int(total_quota * RETRIEVAL_OVER_RATIO)
    print(f"Retrieval quota: {retrieval_quota} (over ratio {RETRIEVAL_OVER_RATIO})")
    show_time(f"{feature_type} retrieval start")
    _, candidate_list = retrieval.equal_quota_retrieval(retrieval_quota)
    show_time(f"{feature_type} retrieval finish, candidates={len(candidate_list)}")
    return candidate_list


def build_candidate_queues(glbinfo, **args) -> CandidateSet:
    budget = int(args["per_round_budget"])
    glbinfo.total_quota = budget
    pos_cand = CandidateSet(list(glbinfo.dataset.features.keys()))
    candidate_pool = set()
    retrieval_pool = set()
    for feature_type in glbinfo.dataset.features:
        retrieved = retrieval_main(glbinfo, feature_type, budget, **args)
        ranked, _ = sorting_main(glbinfo, feature_type, budget, retrieved)
        setattr(pos_cand, f"{feature_type}_list", ranked)
        retrieval_pool.update((int(item[0]), int(item[1])) for item in retrieved)
        candidate_pool.update((int(item[0]), int(item[1])) for item in ranked)
    if args.get("stop") == "certified":
        glbinfo.anytime_candidate_pool = candidate_pool
        glbinfo.anytime_near_pool = retrieval_pool - candidate_pool
    pos_cand.trans_list2cand()
    glbinfo.pos_cand = pos_cand
    return pos_cand
