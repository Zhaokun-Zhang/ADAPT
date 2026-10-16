"""Dispatch the two stopping rules from the paper."""

from ADAPT.stopping_certified import anytime_recall_guarantee
from ADAPT.stopping_heuristic import heuristic_recall_guarantee


def early_stop(glbinfo, **args):
    target = args.get("recall")
    if target is None:
        glbinfo.early_stop = glbinfo.iter_id + 1 >= args["iter_num"]
        return
    if args["stop"] == "certified":
        anytime_recall_guarantee(glbinfo, target_recall=target, **args)
    elif args["stop"] == "heuristic":
        heuristic_recall_guarantee(glbinfo, **args)
    else:
        raise ValueError(f"Unknown stop mode: {args['stop']}")
