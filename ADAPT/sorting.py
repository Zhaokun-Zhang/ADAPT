"""Rank a channel's retrieved pairs by inner product and keep the top budget."""

from typing import List, Tuple

import numpy as np


def sorting_main(glbinfo, feature_type: str, total_quota: int, candidate_list, **args):
    if not candidate_list:
        return [], {"TP": 0, "cmp_num": 0}
    triples = np.array(candidate_list)
    order = np.argsort(-triples[:, 3])
    ranked = triples[order][:, :3].astype(int)

    selected = []
    seen = set()
    for lid, rid, fid in ranked:
        pair = (int(lid), int(rid))
        if pair in seen:
            continue
        seen.add(pair)
        selected.append((pair[0], pair[1], int(fid)))
        if len(selected) >= total_quota:
            break

    actual = {(int(lid), int(rid)) for lid, rid in glbinfo.dataset.joined.values}
    predicted = {(lid, rid) for lid, rid, _ in selected}
    tp = len(predicted & actual)
    print(f"Sort {feature_type}: TP={tp}, queue={len(selected)}")
    return selected, {"TP": tp, "cmp_num": len(selected)}
