"""Heuristic stopping: one saturating exponential plus residual bootstrap.

The verified discovery stream is fit by C(x) = A (1 - exp(-beta x)). Residual
bootstrap supplies an upper quantile of A. The run stops when the resulting
recall estimate meets the target and the estimate has stabilized.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


def _warmup(glbinfo, min_pulls: int, min_iters: int) -> Tuple[bool, str]:
    n_seen = len(getattr(glbinfo, "pull_log", []))
    if n_seen < min_pulls:
        return False, f"warmup: N_seen={n_seen} < {min_pulls}"
    if glbinfo.iter_id < min_iters:
        return False, f"warmup: iter_id={glbinfo.iter_id} < {min_iters}"
    return True, f"warmup passed: N_seen={n_seen}"


def _fit_single(x: np.ndarray, c: np.ndarray):
    def curve(z, amplitude, beta):
        return amplitude * (1.0 - np.exp(-beta * z))

    c_max = float(c[-1])
    guesses = [[c_max * 1.2, 0.001], [c_max * 1.5, 0.0005], [c_max * 2.0, 0.0001]]
    bounds = ([c_max, 0], [c_max * 100, 1.0])
    best_params, best_mse = None, np.inf
    for guess in guesses:
        try:
            params, _ = curve_fit(curve, x, c, p0=guess, bounds=bounds, maxfev=20000)
            pred = curve(x, *params)
            mse = float(np.mean((c - pred) ** 2))
            if mse < best_mse:
                best_params, best_mse = params, mse
        except Exception:
            continue
    return best_params, best_mse


def _bootstrap_upper(x, c, params, n_bootstrap: int, percentile: float) -> float:
    fitted = params[0] * (1.0 - np.exp(-params[1] * x))
    residuals = c - fitted
    samples = []
    for _ in range(n_bootstrap):
        synthetic = fitted + np.random.choice(residuals, size=len(residuals), replace=True)
        boot_params, _ = _fit_single(x, synthetic)
        if boot_params is not None:
            samples.append(float(boot_params[0]))
    if len(samples) < n_bootstrap * 0.5:
        return float(params[0]) * 1.2
    return float(np.percentile(np.asarray(samples), percentile))


def _stable(glbinfo, amplitude, recall_lb, target, rounds: int, epsilon: float) -> Tuple[bool, str]:
    history = glbinfo.recall_certificate_history
    history.append({
        "iter_id": glbinfo.iter_id,
        "C_inf": amplitude,
        "recall_lb": recall_lb,
        "meets_target": recall_lb >= target,
    })
    if len(history) > rounds:
        del history[:-rounds]
    if len(history) < rounds:
        return False, f"stability: {len(history)}/{rounds} rounds accumulated"
    if not all(item["meets_target"] for item in history):
        return False, f"stability: not all of the last {rounds} rounds meet the target"
    values = [item["C_inf"] for item in history]
    max_change = 0.0
    for prev, cur in zip(values, values[1:]):
        max_change = max(max_change, abs(cur - prev) / (prev + 1e-10))
    if max_change > epsilon:
        return False, f"stability: saturation change {max_change:.4f} > {epsilon}"
    return True, f"stability: {rounds} consecutive rounds, saturation converged"


def _record(glbinfo, metrics: dict):
    defaults = {
        "esC_N_seen": 0, "esC_found": 0, "esC_C_inf_point": float("nan"),
        "esC_C_inf_hi": float("nan"), "esC_predicted_recall": float("nan"),
        "esC_recall_lb": float("nan"), "esC_mse": float("nan"),
        "esC_stable": 0, "esC_fire": 0,
    }
    defaults.update(metrics)
    for key, value in defaults.items():
        glbinfo.recorder.add(key, value)


def heuristic_recall_guarantee(glbinfo, **args):
    target = float(args["recall"])
    confidence = float(args.get("recall_confidence", 0.95))
    min_pulls = int(args.get("recall_guarantee_min_pulls", 500))
    min_iters = int(args.get("recall_guarantee_min_iters", 2))
    epsilon = float(args.get("recall_guarantee_epsilon", 0.05))
    n_bootstrap = int(args.get("recall_guarantee_n_bootstrap", 200))
    rounds = int(args.get("recall_guarantee_K", 1))
    percentile = confidence * 100
    glbinfo.early_stop = False

    ok, message = _warmup(glbinfo, min_pulls, min_iters)
    if not ok:
        print(f"[stop][heuristic] {message}")
        _record(glbinfo, {"esC_N_seen": len(glbinfo.pull_log)})
        return

    rewards = np.array([record[2] for record in glbinfo.pull_log], dtype=np.float64)
    cumulative = np.cumsum(rewards)
    x = np.arange(len(cumulative), dtype=np.float64)
    found = int(cumulative[-1])
    params, mse = _fit_single(x, cumulative)
    if params is None:
        print("[stop][heuristic] single-exponential fit failed")
        _record(glbinfo, {"esC_N_seen": len(x), "esC_found": found})
        return

    amplitude = float(params[0])
    amplitude_hi = _bootstrap_upper(x, cumulative, params, n_bootstrap, percentile)
    recall_lb = found / amplitude_hi if amplitude_hi > 0 else 0.0
    predicted = found / amplitude if amplitude > 0 else 0.0
    stable, stability_msg = _stable(glbinfo, amplitude_hi, recall_lb, target, rounds, epsilon)
    fire = bool(stable and recall_lb >= target)
    glbinfo.early_stop = fire

    print("=" * 72)
    print("Heuristic stopping (single saturating exponential)")
    print(f"  target recall={target:.4f}  confidence={confidence:.2f}")
    print(f"  N_seen={len(x)}  found={found}  A={amplitude:.1f}  A_hi={amplitude_hi:.1f}")
    print(f"  predicted recall={predicted:.4f}  recall lower bound={recall_lb:.4f}  mse={mse:.6f}")
    print(f"  {stability_msg}")
    print(f"  decision: {'STOP' if fire else 'CONTINUE'}")
    print("=" * 72)
    _record(glbinfo, {
        "esC_N_seen": len(x), "esC_found": found,
        "esC_C_inf_point": amplitude, "esC_C_inf_hi": amplitude_hi,
        "esC_predicted_recall": predicted, "esC_recall_lb": recall_lb,
        "esC_mse": mse, "esC_stable": int(stable), "esC_fire": int(fire),
    })
