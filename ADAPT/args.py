import argparse

from ADAPT.paths import FEATURE_TYPES


def get_exc_args() -> dict:
    parser = argparse.ArgumentParser(description="Adapt semantic join")
    parser.add_argument("--dataset", default="MovieActor", type=str,
                        help="directory name under data/")
    parser.add_argument("--per_round_budget", "-T", default=500, type=int,
                        help="LLM verification budget of one iteration")
    parser.add_argument("--iter_num", default=10, type=int,
                        help="maximum number of iterations")
    parser.add_argument("--stop", default="certified", choices=["certified", "heuristic"],
                        help="certified: anytime-valid audit; heuristic: saturating exponential")
    parser.add_argument("--quality", default=None, type=float,
                        help="target F1 in (0, 1]. Omit to run every iteration.")
    parser.add_argument("--confidence", default=0.9, type=float,
                        help="confidence level 1-delta for the stopping rule")
    parser.add_argument("--seed", default=114514, type=int)
    parser.add_argument("--embedding_model_name", default="e5", type=str)
    parser.add_argument("--cuda", default=0, type=int,
                        help="cuda device id; negative values force CPU")
    parser.add_argument("--llm_mode", default="cache",
                        choices=["simulation", "cache", "normal"])
    parser.add_argument("--llm_max_workers", default=16, type=int)
    parser.add_argument("--profiling_sample_k", default=10, type=int)
    parser.add_argument("--redirect_log", action="store_true")

    parser.add_argument("--recall_guarantee_min_pulls", default=500, type=int)
    parser.add_argument("--recall_guarantee_min_iters", default=2, type=int)
    parser.add_argument("--recall_guarantee_epsilon", default=0.05, type=float,
                        help="relative change of the saturation level required to stop")
    parser.add_argument("--recall_guarantee_n_bootstrap", default=200, type=int)
    parser.add_argument("--recall_guarantee_K", default=1, type=int,
                        help="consecutive stable rounds required by the heuristic stopper")

    parser.add_argument("--anytime_audit_ratio", default=1.0, type=float,
                        help="extra certified-audit budget as a fraction of -T")
    parser.add_argument("--anytime_min_audit", default=1, type=int)
    parser.add_argument("--anytime_uniform_mix", default=0.50, type=float)
    parser.add_argument("--anytime_prior_strength", default=2.0, type=float)
    parser.add_argument("--anytime_candidate_prior", default=0.20, type=float)
    parser.add_argument("--anytime_near_prior", default=0.05, type=float)
    parser.add_argument("--anytime_deep_prior", default=0.01, type=float)
    parser.add_argument("--anytime_deep_max_tries", default=100000, type=int)
    parser.add_argument("--anytime_check_every", default=1, type=int)

    parser.add_argument("--dl_hidden_ratio", default=2.0, type=float)
    parser.add_argument("--dl_dropout", default=0.1, type=float)
    parser.add_argument("--dl_epochs", default=200, type=int)
    parser.add_argument("--dl_lr", default=1e-4, type=float)
    parser.add_argument("--dl_weight_decay", default=1e-4, type=float)
    parser.add_argument("--dl_batch_size", default=256, type=int)
    parser.add_argument("--dl_temperature", default=0.05, type=float)
    parser.add_argument("--dl_neg_k", default=32, type=int)
    parser.add_argument("--dl_hard_ratio", default=0.5, type=float)
    parser.add_argument("--dl_hard_pool", default=64, type=int)
    parser.add_argument("--dl_min_pos", default=5, type=int)
    parser.add_argument("--dl_max_drift", default=0.3, type=float)
    parser.add_argument("--dl_val_min", default=20, type=int)
    parser.add_argument("--dl_val_ratio", default=0.2, type=float)
    parser.add_argument("--dl_patience", default=20, type=int)

    args = parser.parse_args()
    args = vars(args)
    if args["per_round_budget"] <= 0:
        parser.error("--per_round_budget / -T must be a positive integer.")
    if args["quality"] is not None and not 0.0 < args["quality"] <= 1.0:
        parser.error("--quality must be in (0, 1].")
    if not 0.0 < args["confidence"] < 1.0:
        parser.error("--confidence must be in (0, 1).")

    # Returned pairs are LLM-verified, so oracle precision is 1 and
    # F1 = 2R/(1+R) is equivalent to recall = F1/(2-F1).
    if args["quality"] is None:
        args["recall"] = None
    else:
        args["recall"] = args["quality"] / (2.0 - args["quality"])
    args["recall_confidence"] = args["confidence"]
    args["features"] = list(FEATURE_TYPES)
    args["mark"] = (
        f"{args['dataset']}|T{args['per_round_budget']}|{args['stop']}|"
        f"q{args['quality']}|c{args['confidence']}|{args['llm_mode']}"
    )

    try:
        import torch
        cuda_ok = args["cuda"] >= 0 and torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    args["device"] = f"cuda:{args['cuda']}" if cuda_ok else "cpu"

    import random
    import numpy as np
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    try:
        import torch
        torch.manual_seed(args["seed"])
    except Exception:
        pass

    print("\n" + "-" * 25 + " Parameters " + "-" * 25)
    for key, value in args.items():
        print(f"{key:>30} : {value}")
    print("-" * 62)
    return args
