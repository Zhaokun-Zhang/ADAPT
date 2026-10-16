"""Anytime-valid recall certification for Semantic Join (``--early_stop D``).

This module deliberately separates *discovery* from *certification*:

* The existing retrieval/allocation code may select oracle queries with arbitrary bias.
  Those labels increase the number of discovered positives, but are never treated as a
  probability sample.
* This module draws an additional audit sample with a known, history-measurable
  probability.  A prediction-powered inverse-propensity residual corrects any error in
  the stratum predictor.

Let ``P`` be the fixed number of oracle-positive pairs in ``L x R``.  Immediately before
audit draw t, let K be the labelled set, F its positive count, U the unlabelled set,
f_t(x) a predictor fixed before the draw, and q_t the audit distribution on U.  For
I_t ~ q_t, the observation

    Z_t = F + sum_{x in U} f_t(x) + (Y[I_t] - f_t(I_t)) / q_t(I_t)

satisfies ``E[Z_t | history] = P``.  Hence candidate pools, predictors and sampling
weights may all change between draws without changing the estimand.  A one-sided
betting e-process is inverted to obtain a time-uniform upper confidence bound for P;
``F / P_ucb`` is therefore a time-uniform lower confidence bound for recall.

Validity assumptions
--------------------
1. Oracle labels define a fixed binary predicate during a run (cache/deterministic
   decoding is recommended for an LLM oracle).
2. The partition, predictor and q_t are fixed before observing the current audit label.
3. Every currently unlabelled pair has positive audit probability.  The uniform part of
   the robust active/uniform mixture enforces this condition.

Predictions and strata affect efficiency only.  They may be arbitrarily inaccurate
without invalidating the certificate.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, MutableMapping, Optional, Set, Tuple

import numpy as np


Pair = Tuple[int, int]
STRATA = ("candidate", "near_tail", "deep_tail")
_STATE_VERSION = 1


def _validate_config(target_recall: float, args: MutableMapping) -> Dict[str, float]:
    """Validate Plan-D parameters and return their normalized numeric values."""
    tau = float(target_recall)
    confidence = float(args.get("recall_confidence", 0.95))
    audit_ratio = float(args.get("anytime_audit_ratio", 1.0))
    uniform_mix = float(args.get("anytime_uniform_mix", 0.50))
    prior_strength = float(args.get("anytime_prior_strength", 2.0))

    if not 0.0 < tau <= 1.0:
        raise ValueError("--recall must be in (0, 1].")
    if not 0.0 < confidence < 1.0:
        raise ValueError("--recall_confidence must be in (0, 1).")
    if audit_ratio < 0.0:
        raise ValueError("--anytime_audit_ratio must be non-negative.")
    if not 0.0 < uniform_mix <= 1.0:
        raise ValueError("--anytime_uniform_mix must be in (0, 1].")
    if prior_strength <= 0.0:
        raise ValueError("--anytime_prior_strength must be positive.")

    priors = {
        "candidate": float(args.get("anytime_candidate_prior", 0.20)),
        "near_tail": float(args.get("anytime_near_prior", 0.05)),
        "deep_tail": float(args.get("anytime_deep_prior", 0.01)),
    }
    if any(not 0.0 <= value <= 1.0 for value in priors.values()):
        raise ValueError("Plan-D stratum priors must be in [0, 1].")

    return {
        "tau": tau,
        "confidence": confidence,
        "alpha": 1.0 - confidence,
        "audit_ratio": audit_ratio,
        "uniform_mix": uniform_mix,
        "prior_strength": prior_strength,
        **{f"prior_{key}": value for key, value in priors.items()},
    }


def _empty_stratum_stats() -> Dict[str, Dict[str, float]]:
    """Create audit-only sufficient statistics used by the nuisance predictor."""
    return {
        name: {"n": 0, "sum_y": 0.0, "sum_resid_sq": 0.0}
        for name in STRATA
    }


def _load_or_initialize_state(glbinfo, alpha: float, seed: int) -> Dict:
    """Load cross-round state from ``glbinfo`` or initialize a fresh e-process.

    The state is attached dynamically so GlobalInfo does not need to be modified.  A
    confidence-level or population-shape change starts a new statistical experiment.
        ├── historical DR observations
        │       ├── betting e-process
        │       ├── stratified audit statistics
        │       └── previous P_UCB
    """
    shape = tuple(int(v) for v in glbinfo.llm_joined.shape)
    state = getattr(glbinfo, "_anytime_guarantee_state", None)
    reset = (
        not isinstance(state, dict)
        or state.get("version") != _STATE_VERSION
        or tuple(state.get("shape", ())) != shape
        or not np.isclose(float(state.get("alpha", -1.0)), alpha)
    )
    if reset:
        population_size = int(shape[0]) * int(shape[1])
        state = {
            "version": _STATE_VERSION,
            "shape": shape,
            "alpha": float(alpha),
            "rng": np.random.default_rng(int(seed) + 0xD00D),
            "records": [],
            "stratum_stats": _empty_stratum_stats(),
            "audit_count": 0,
            "last_upper": population_size,
            "last_recall_lcb": 0.0,
        }
        glbinfo._anytime_guarantee_state = state
    return state


def _pairs_from_feature_lists(candidate_set) -> Set[Pair]:
    """Deduplicate pair identities across every feature queue."""
    pairs: Set[Pair] = set()
    if candidate_set is None:
        return pairs
    for feature_type in getattr(candidate_set, "feature_type_list", []):
        for item in getattr(candidate_set, f"{feature_type}_list", []) or []:
            if len(item) >= 2:
                pairs.add((int(item[0]), int(item[1])))
    return pairs


def _pairs_from_union_dict(candidate_set) -> Set[Pair]:
    """Read a CandidateSet union dictionary without assuming it was initialized."""
    pairs: Set[Pair] = set()
    if candidate_set is None:
        return pairs
    for lid, rids in (getattr(candidate_set, "union_cand", {}) or {}).items():
        pairs.update((int(lid), int(rid)) for rid in rids)
    return pairs


def _filter_unlabelled(pairs: Iterable[Pair], labels) -> Set[Pair]:
    """Keep only pairs that are still in the finite population's unknown set."""
    return {pair for pair in pairs if labels[pair[0], pair[1]] < 0}


def _build_dynamic_strata(glbinfo) -> Dict:
    """Build an exact disjoint partition of the current unlabelled universe.

    Preferred inputs are snapshots saved by retrieval_finetuning:
      candidate = sorted pairs that entered a feature queue;
      near_tail = over-retrieved pairs that did not enter any final queue.

    For old caches/configurations without those snapshots, the method falls back to the
    current candidate set and the historical-candidate difference.  Deep-tail remains
    the implicit complement and is never materialized.
    """
    labels = glbinfo.llm_joined
    m, n = (int(v) for v in labels.shape)
    labelled_count = int(labels.count_labeled())
    unknown_count = int(m) * int(n) - labelled_count

    candidate_snapshot = getattr(glbinfo, "anytime_candidate_pool", None)
    near_snapshot = getattr(glbinfo, "anytime_near_pool", None)

    if candidate_snapshot is None:
        candidate = _pairs_from_feature_lists(getattr(glbinfo, "pos_cand", None))
    else:
        candidate = {(int(lid), int(rid)) for lid, rid in candidate_snapshot}

    if near_snapshot is None:
        historical = _pairs_from_union_dict(getattr(glbinfo, "all_pos_cand", None))
        near_tail = historical - candidate
    else:
        near_tail = {(int(lid), int(rid)) for lid, rid in near_snapshot}

    candidate = _filter_unlabelled(candidate, labels)
    # Canonical precedence resolves overlap across feature queues and snapshots.
    near_tail = _filter_unlabelled(near_tail, labels) - candidate
    deep_count = unknown_count - len(candidate) - len(near_tail)
    if deep_count < 0:
        raise RuntimeError("Plan-D strata are not a disjoint cover of the unknown space.")

    return {
        "candidate": candidate,
        "near_tail": near_tail,
        "deep_tail_count": int(deep_count),
        "unknown_count": int(unknown_count),
        "m": m,
        "n": n,
    }


def _stratum_sizes(regions: Dict) -> Dict[str, int]:
    """Return current sizes, including the implicit deep-tail complement."""
    return {
        "candidate": len(regions["candidate"]),
        "near_tail": len(regions["near_tail"]),
        "deep_tail": int(regions["deep_tail_count"]),
    }


def _build_stratum_predictors(state: Dict, config: Dict[str, float]) -> Dict[str, float]:
    """Build predictable, audit-only Beta-smoothed positive-rate predictors.

    These values are nuisance predictions, not assumptions.  The IP residual makes the
    total estimator unbiased even when every value below is wrong.
    """
    strength = config["prior_strength"]
    predictions: Dict[str, float] = {}
    for name in STRATA:
        stats = state["stratum_stats"][name]
        prior = config[f"prior_{name}"]
        predictions[name] = float(
            (prior * strength + stats["sum_y"]) / (strength + stats["n"])
        )
    return predictions


def _compute_robust_sampling_weights(
    sizes: Dict[str, int], predictions: Dict[str, float], state: Dict,
    uniform_mix: float,
) -> Dict[str, float]:
    """Compute robust active/uniform stratum masses.

    The active component is a plug-in Neyman allocation ``N_h * sqrt(v_h)``.  It is
    mixed with pair-uniform sampling, so every unknown pair has probability at least
    ``uniform_mix / |U|`` regardless of uncertainty-model quality.
    """
    total = int(sum(sizes.values()))
    if total <= 0:
        return {name: 0.0 for name in STRATA}

    active_raw: Dict[str, float] = {}
    for name in STRATA:
        size = sizes[name]
        if size <= 0:
            active_raw[name] = 0.0
            continue
        stats = state["stratum_stats"][name]
        if stats["n"] > 0:
            residual_second_moment = stats["sum_resid_sq"] / stats["n"]
        else:
            p = predictions[name]
            residual_second_moment = p * (1.0 - p)
        # A small floor prevents a zero estimated variance from disabling active audit.
        variance_proxy = max(float(residual_second_moment), 1e-6)
        active_raw[name] = float(size) * float(np.sqrt(variance_proxy))

    active_total = float(sum(active_raw.values()))
    if active_total <= 0.0:
        active = {name: sizes[name] / total for name in STRATA}
    else:
        active = {name: active_raw[name] / active_total for name in STRATA}

    weights = {
        name: uniform_mix * (sizes[name] / total) + (1.0 - uniform_mix) * active[name]
        for name in STRATA
    }
    # Remove numerical mass on empty strata and renormalize exactly.
    weights = {name: (weights[name] if sizes[name] > 0 else 0.0) for name in STRATA}
    normalizer = float(sum(weights.values()))
    return {name: weights[name] / normalizer for name in STRATA}


def _draw_stratum(weights: Dict[str, float], rng: np.random.Generator) -> str:
    """Draw one nonempty stratum according to its logged probability mass."""
    names = [name for name in STRATA if weights[name] > 0.0]
    probs = np.asarray([weights[name] for name in names], dtype=float)
    probs /= probs.sum()
    return names[int(rng.choice(len(names), p=probs))]


def _sample_explicit_stratum(
    pairs: Set[Pair], rng: np.random.Generator,
) -> Pair:
    """Uniformly sample a pair from a materialized candidate/near-tail set."""
    if not pairs:
        raise RuntimeError("Cannot sample from an empty explicit stratum.")
    # Candidate regions are small compared with L x R; conversion preserves exact
    # uniformity and avoids imposing ordering requirements on CandidateSet.
    pair_list = tuple(pairs)
    return pair_list[int(rng.integers(0, len(pair_list)))]


def _sample_deep_tail(
    regions: Dict, labels, rng: np.random.Generator, max_tries: int,
) -> Optional[Pair]:
    """Uniformly sample the implicit deep tail by rejection over ``L x R``.

    Conditional on acceptance, a uniform Cartesian-product draw is exactly uniform on
    the complement.  Returning None merely skips this audit observation; it never
    records a draw with an incorrect propensity.
    """
    if regions["deep_tail_count"] <= 0:
        return None
    candidate = regions["candidate"]
    near_tail = regions["near_tail"]
    for _ in range(max_tries):
        lid = int(rng.integers(0, regions["m"]))
        rid = int(rng.integers(0, regions["n"]))
        pair = (lid, rid)
        if pair in candidate or pair in near_tail or labels[lid, rid] >= 0:
            continue
        return pair
    return None


def _draw_audit_pair(
    regions: Dict, sizes: Dict[str, int], weights: Dict[str, float], labels,
    rng: np.random.Generator, max_tries: int,
) -> Optional[Tuple[Pair, str, float]]:
    """Draw an audit pair and return ``(pair, stratum, pair_probability)``."""
    stratum = _draw_stratum(weights, rng)
    if stratum == "candidate":
        pair = _sample_explicit_stratum(regions["candidate"], rng)
    elif stratum == "near_tail":
        pair = _sample_explicit_stratum(regions["near_tail"], rng)
    else:
        pair = _sample_deep_tail(regions, labels, rng, max_tries)
        if pair is None:
            return None
    pair_probability = float(weights[stratum] / sizes[stratum])
    return pair, stratum, pair_probability


def _prediction_total(
    found_positive: int, sizes: Dict[str, int], predictions: Dict[str, float],
) -> float:
    """Compute ``F + sum_U f(x)`` using only three aggregate stratum counts."""
    return float(found_positive) + sum(
        float(sizes[name]) * predictions[name] for name in STRATA
    )


def _dr_support_bounds(
    prediction_total: float, sizes: Dict[str, int], predictions: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, float]:
    """Compute predictable support bounds for the next DR observation."""
    lower_corrections: List[float] = []
    upper_corrections: List[float] = []
    for name in STRATA:
        if sizes[name] <= 0:
            continue
        pair_probability = weights[name] / sizes[name]
        lower_corrections.append(predictions[name] / pair_probability)
        upper_corrections.append((1.0 - predictions[name]) / pair_probability)
    lower = prediction_total - max(lower_corrections, default=0.0)
    upper = prediction_total + max(upper_corrections, default=0.0)
    return float(lower), float(upper)


def _query_audit_label(glbinfo, pair: Pair, args: MutableMapping, joined_set=None) -> int:
    """Verify one audit pair with the LLM and persist the label in llm_joined."""
    from ADAPT.allocation import verify_pair
    return int(verify_pair(glbinfo, pair[0], pair[1]))


def _record_dr_observation(
    state: Dict, glbinfo, pair: Pair, stratum: str, probability: float,
    label: int, prediction: float, prediction_total: float,
    support: Tuple[float, float],
) -> float:
    """Append one DR martingale observation and update audit-only statistics."""
    z_value = float(prediction_total + (float(label) - prediction) / probability)
    lower, upper = support
    tolerance = 1e-8 * max(1.0, abs(lower), abs(upper), abs(z_value))
    if z_value < lower - tolerance or z_value > upper + tolerance:
        raise RuntimeError("DR observation lies outside its predictable support.")

    record = {
        "z": z_value,
        "lower": float(lower),
        "upper": float(upper),
        "q": float(probability),
        "label": int(label),
        "prediction": float(prediction),
        "pair": (int(pair[0]), int(pair[1])),
        "stratum": stratum,
        "iter_id": int(glbinfo.iter_id),
    }
    state["records"].append(record)
    state["audit_count"] += 1

    stats = state["stratum_stats"][stratum]
    stats["n"] += 1
    stats["sum_y"] += float(label)
    stats["sum_resid_sq"] += float((float(label) - prediction) ** 2)

    # Audit observations are deliberately not written to pull_log: Plan A assumes a
    # relevance-ordered discovery stream.  audit_log remains a descriptive log only;
    # q and all proof-critical fields live in this module's state.
    glbinfo.audit_log.append(
        (int(pair[0]), int(pair[1]), int(label), int(glbinfo.iter_id), "D")
    )
    return z_value


def _remove_audited_pair(regions: Dict, pair: Pair, stratum: str) -> None:
    """Remove the newly labelled pair so the next draw is without replacement."""
    if stratum == "candidate":
        regions["candidate"].remove(pair)
    elif stratum == "near_tail":
        regions["near_tail"].remove(pair)
    else:
        regions["deep_tail_count"] -= 1
    regions["unknown_count"] -= 1


def _log_mixture_evalue(records: List[Dict], p_null: int) -> float:
    """Evaluate a valid one-sided betting e-process for ``H0: P >= p_null``.

    For predictable upper support B_t and betting fraction c in (0,1),

        1 + c * (p_null - Z_t) / (B_t - p_null)

    is nonnegative and has conditional expectation at most one under H0.  An equal
    mixture over several c values is again an e-process and is less sensitive to one
    arbitrary bet size.  The expression is monotone in p_null, enabling integer
    bisection over the unknown positive count.
    """
    if not records:
        return 0.0  # log(1)
    z = np.fromiter((float(record["z"]) for record in records), dtype=float)
    upper = np.fromiter((float(record["upper"]) for record in records), dtype=float)
    p = float(p_null)
    tol = 1e-12 * np.maximum(1.0, np.abs(upper))

    # If p exceeds a predictable support upper bound, H0 is impossible.  At equality,
    # E[Z]=p with Z<=p requires Z=p almost surely; observing Z<p also rejects exactly.
    if np.any(p > upper + tol):
        return float("inf")
    equal = np.abs(upper - p) <= tol
    if np.any(equal & (z < p - tol)):
        return float("inf")

    active = ~equal
    fractions = np.asarray((0.05, 0.10, 0.25, 0.50, 0.75), dtype=float)
    component_logs = np.zeros(len(fractions), dtype=float)
    if np.any(active):
        denominator = upper[active] - p
        delta = p - z[active]
        for index, fraction in enumerate(fractions):
            factors = 1.0 + fraction * delta / denominator
            # Mathematically factors >= 1-fraction.  This guard only absorbs floating
            # point noise at the support boundary.
            factors = np.maximum(factors, np.finfo(float).tiny)
            component_logs[index] = float(np.log(factors).sum())

    # Stable log(mean(exp(component_logs))).
    maximum = float(np.max(component_logs))
    return maximum + float(np.log(np.exp(component_logs - maximum).mean()))


def _invert_eprocess_to_upper_bound(
    state: Dict, found_positive: int, population_size: int, alpha: float,
) -> int:
    """Invert the monotone e-process tests into an integer upper CS for P.

    The running minimum forms the usual running-intersection confidence sequence, so
    the reported upper bound never widens as more audit evidence arrives.
    """
    previous = int(state.get("last_upper", population_size))
    search_high = max(int(found_positive), min(previous, int(population_size)))
    threshold = float(np.log(1.0 / alpha))

    # Most observations do not tighten the bound; this fast path avoids a bisection.
    if _log_mixture_evalue(state["records"], search_high) < threshold:
        return search_high

    low = int(found_positive)
    high = search_high
    while low < high:
        middle = (low + high) // 2
        if _log_mixture_evalue(state["records"], middle) >= threshold:
            high = middle
        else:
            low = middle + 1

    # ``low`` is the first rejected integer null.  Every non-rejected possible P is at
    # most low-1.  Clamping to F handles the alpha-probability failure/numerical edge.
    return max(int(found_positive), int(low) - 1)


def _compute_recall_lower_bound(found_positive: int, positive_upper: int) -> float:
    """Convert an upper confidence bound for P into a lower bound for recall."""
    if positive_upper <= 0:
        return 1.0 if found_positive <= 0 else 0.0
    return float(min(1.0, found_positive / max(found_positive, positive_upper)))


def _audit_budget(glbinfo, config: Dict[str, float], args: MutableMapping) -> int:
    """Return the *additional* per-round audit quota requested by the user."""
    total_quota = max(0, int(getattr(glbinfo, "total_quota", 0)))
    override = args.get("anytime_audit_budget_override")
    if override is None:
        budget = int(round(total_quota * config["audit_ratio"]))
        minimum = max(0, int(args.get("anytime_min_audit", 1)))
        if total_quota > 0 and config["audit_ratio"] > 0.0:
            budget = max(budget, minimum)
    else:
        budget = max(0, int(override))
    glbinfo.exploit_budget = total_quota
    glbinfo.audit_budget = budget
    glbinfo.round_oracle_budget = total_quota + budget
    return budget


def _record_diagnostics(
    glbinfo, state: Dict, sizes: Dict[str, int], round_audits: int,
    round_positives: int, positive_upper: int, recall_lcb: float, fire: bool,
) -> None:
    """Persist one compact diagnostic row for the current Semantic Join iteration."""
    for key, value in [
        ("esD_recall_lcb", recall_lcb),
        ("esD_P_found", glbinfo.llm_joined.count_positive()),
        ("esD_P_ucb", positive_upper),
        ("esD_audit_total", state["audit_count"]),
        ("esD_audit_round", round_audits),
        ("esD_audit_pos_round", round_positives),
        ("esD_N_candidate", sizes["candidate"]),
        ("esD_N_near", sizes["near_tail"]),
        ("esD_N_deep", sizes["deep_tail"]),
        ("esD_fire", int(fire)),
    ]:
        glbinfo.recorder.add(key, value)


def anytime_recall_guarantee(glbinfo, target_recall: Optional[float] = None, **args):
    """Run one round of Plan-D audit and update ``glbinfo.early_stop``.

    The function is intentionally orchestration-only: every statistical or systems
    operation is delegated to a single-purpose helper above.  Audit observations and
    betting wealth persist across Semantic Join iterations.
    """
    if target_recall is None:
        target_recall = args.get("recall")
    if target_recall is None:
        raise ValueError("Plan D requires --recall.")

    config = _validate_config(float(target_recall), args)  # recall, confidence, audit ratio
    state = _load_or_initialize_state(  # restore or initialize cross-round state
        glbinfo, alpha=config["alpha"], seed=int(args.get("seed", 0))
    )
    glbinfo.early_stop = False

    regions = _build_dynamic_strata(glbinfo)  # three disjoint strata over the unlabeled space
    initial_sizes = _stratum_sizes(regions)
    population_size = int(regions["m"]) * int(regions["n"])
    budget = _audit_budget(glbinfo, config, args)  # additional audit budget
    max_tries = max(1, int(args.get("anytime_deep_max_tries", 100000)))
    check_every = max(1, int(args.get("anytime_check_every", 1)))
    audit_policy = str(args.get("anytime_audit_policy", "active"))
    if audit_policy not in {"active", "uniform"}:
        raise ValueError("--anytime_audit_policy must be 'active' or 'uniform'.")

    reward_mode = str(args.get("anytime_reward_mode", "oracle"))
    joined_set = None
    if reward_mode == "gt":
        joined_set = {
            (int(lid), int(rid)) for lid, rid in glbinfo.dataset.joined.values
        }
    elif reward_mode != "oracle":
        raise ValueError("--anytime_reward_mode must be 'oracle' or 'gt'.")

    round_audits = 0
    round_positives = 0
    positive_upper = max(
        glbinfo.llm_joined.count_positive(),
        min(int(state["last_upper"]), population_size),
    )
    recall_lcb = _compute_recall_lower_bound(
        glbinfo.llm_joined.count_positive(), positive_upper
    )

    # The previous audit CS remains valid between audit draws.  New exploitation
    # positives can therefore trigger stopping even before spending this round's audit.
    if recall_lcb < config["tau"]:
        for audit_index in range(budget):
            sizes = _stratum_sizes(regions)
            if sum(sizes.values()) <= 0:
                positive_upper = glbinfo.llm_joined.count_positive()
                recall_lcb = 1.0
                break

            if audit_policy == "uniform":
                # Uniform-CS baseline: no stratified nuisance prediction and exactly
                # pair-uniform sampling over the current unknown population.
                predictions = {name: 0.0 for name in STRATA}
                unknown_total = float(sum(sizes.values()))
                weights = {
                    name: (sizes[name] / unknown_total if unknown_total > 0.0 else 0.0)
                    for name in STRATA
                }
            else:
                predictions = _build_stratum_predictors(state, config)  # stratum match rates from past audits; a wrong predictor does not break the guarantee
                weights = _compute_robust_sampling_weights(  # mix active and uniform sampling
                    sizes, predictions, state, config["uniform_mix"]  # q = w / N stays bounded because the uniform component is present, not only Neyman allocation
                )
            draw = _draw_audit_pair(  # sample one pair and its stratum and probability
                regions, sizes, weights, glbinfo.llm_joined,
                state["rng"], max_tries,
            )
            if draw is None:
                print("[early_stop][D] deep-tail rejection sampler exhausted; "
                      "skipping the remaining audit quota without adding evidence.")
                break

            pair, stratum, probability = draw
            found_before = int(glbinfo.llm_joined.count_positive())
            prediction_total = _prediction_total(found_before, sizes, predictions)  # predicted match count from the three strata, without enumerating L x R
            support = _dr_support_bounds(  # predictable bounds of the DR observation
                prediction_total, sizes, predictions, weights
            )
            label = _query_audit_label(glbinfo, pair, args, joined_set=joined_set)
            _record_dr_observation(
                state, glbinfo, pair, stratum, probability, label,
                predictions[stratum], prediction_total, support,
            )
            _remove_audited_pair(regions, pair, stratum)
            round_audits += 1
            round_positives += int(label)

            # Computing the bound does not affect validity.  check_every=1 exposes the
            # full per-label anytime behavior; larger values trade responsiveness for
            # less numerical work while retaining the same underlying e-process.
            should_check = ((audit_index + 1) % check_every == 0) or (label == 1)
            if should_check:
                found_now = int(glbinfo.llm_joined.count_positive())
                positive_upper = _invert_eprocess_to_upper_bound(
                    state, found_now, population_size, config["alpha"]
                )
                state["last_upper"] = positive_upper
                recall_lcb = _compute_recall_lower_bound(found_now, positive_upper)
                if recall_lcb >= config["tau"]:
                    break

    # Ensure the final partial check interval is reflected in the reported certificate.
    found_final = int(glbinfo.llm_joined.count_positive())
    if round_audits > 0:
        positive_upper = _invert_eprocess_to_upper_bound(
            state, found_final, population_size, config["alpha"]
        )
        state["last_upper"] = positive_upper
    recall_lcb = _compute_recall_lower_bound(found_final, positive_upper)
    state["last_recall_lcb"] = recall_lcb

    fire = bool(recall_lcb >= config["tau"])
    glbinfo.early_stop = fire
    final_sizes = _stratum_sizes(regions)
    _record_diagnostics(
        glbinfo, state, initial_sizes, round_audits, round_positives,
        positive_upper, recall_lcb, fire,
    )

    glbinfo.last_recall_certificate = {
        "method": "D",
        "audit_policy": audit_policy,
        "confidence": config["confidence"],
        "target_recall": config["tau"],
        "positive_found": found_final,
        "positive_upper": int(positive_upper),
        "recall_lcb": float(recall_lcb),
        "audit_total": int(state["audit_count"]),
        "audit_round": int(round_audits),
        "audit_positive_round": int(round_positives),
        "initial_strata": initial_sizes,
        "remaining_strata": final_sizes,
        "fire": fire,
    }
    glbinfo.recall_certificate_history.append(dict(glbinfo.last_recall_certificate))

    print("============= Recall Guarantee (Plan D: anytime DR betting) =============")
    print(f"  tau={config['tau']}  confidence={config['confidence']}  "
          f"audit={round_audits}/{budget} (additional quota)  policy={audit_policy}")
    print(f"  strata(before): C={initial_sizes['candidate']}  "
          f"T={initial_sizes['near_tail']}  D={initial_sizes['deep_tail']}")
    print(f"  P_found={found_final}  P_ucb={positive_upper}  "
          f"recall_LCB={recall_lcb:.6g}")
    print(f"  -> {'STOP' if fire else 'CONTINUE'}")
    print("========================================================================")
    return dict(glbinfo.last_recall_certificate)


__all__ = ["anytime_recall_guarantee"]
