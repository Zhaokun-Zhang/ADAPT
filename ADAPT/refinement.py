"""Cross-channel residual-MLP refinement of driving-side embeddings.

Verified record pairs are mapped to one feature embedding per channel by
argmax cosine similarity against the driven record. Original embeddings stay
fixed, so each (record, channel) mapping is computed once and reused. Later
rounds only map pairs that were verified since the previous round.
"""

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_EMPTY = np.empty(0, dtype=np.int64)


def _cfg(**args) -> dict:
    defaults = dict(
        dl_hidden_ratio=2.0,
        dl_dropout=0.1,
        dl_epochs=200,
        dl_lr=1e-4,
        dl_weight_decay=1e-4,
        dl_batch_size=256,
        dl_temperature=0.05,
        dl_neg_k=32,
        dl_hard_ratio=0.5,
        dl_hard_pool=64,
        dl_min_pos=5,
        dl_max_drift=0.3,
        dl_val_min=20,
        dl_val_ratio=0.2,
        dl_patience=20,
    )
    cfg = dict(defaults)
    for key in cfg:
        if args.get(key) is not None:
            cfg[key] = args[key]
    return cfg


class MLPProjector(nn.Module):
    """Three-layer residual MLP. The last layer starts at zero, so g(x)=normalize(x)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.net(x), p=2, dim=-1)


def update_pair_feature_cache(rtr) -> Dict[Tuple[int, int], int]:
    """Map newly verified (Lid, Rid) pairs onto this channel's original features.

    The chosen feature is the one whose original embedding is most similar to
    the driven record. Because those embeddings never change, a stored fid is
    valid for every later round.
    """
    cache: Dict[Tuple[int, int], int] = rtr.feat_dict.setdefault("pair_fid", {})
    labeled = rtr.llm_joined.get_labeled_pairs()
    pending = defaultdict(list)
    for lid, rid in labeled:
        key = (int(lid), int(rid))
        if key not in cache:
            pending[key[0]].append(key[1])
    if not pending:
        print(f"[pair-map] feature={rtr.feature_type} new=0 cached={len(cache)}")
        return cache

    feat_emb = rtr.feat_emb
    remb = rtr.Remb
    n_new = 0
    for lid, rids in pending.items():
        fids = np.asarray(rtr.Lid2fids.get(lid, []), dtype=np.int64)
        if len(fids) == 0:
            continue
        scores = feat_emb[fids] @ remb[np.asarray(rids, dtype=np.int64)].T
        best = np.argmax(scores, axis=0)
        for rid, index in zip(rids, best):
            cache[(lid, int(rid))] = int(fids[int(index)])
            n_new += 1
    print(f"[pair-map] feature={rtr.feature_type} new={n_new} cached={len(cache)}")
    return cache


def _anchor_fids(rtr, pairs: np.ndarray) -> np.ndarray:
    cache = rtr.feat_dict["pair_fid"]
    return np.asarray([cache[(int(lid), int(rid))] for lid, rid in pairs], dtype=np.int64)


def _hard_neg_pool(rtr, anchor_fid: np.ndarray, pool: int) -> np.ndarray:
    if pool <= 0:
        return None
    unique, inverse = np.unique(anchor_fid, return_inverse=True)
    topk = min(pool, int(rtr.Remb.shape[0]))
    _, rid_mat = rtr.Rindex.search(
        np.ascontiguousarray(rtr.feat_emb[unique], dtype=np.float32), topk)
    return rid_mat[inverse]


def _neg_rids(lid: int, llm_joined, neg_k: int, rng: np.random.Generator,
              hard_pool: np.ndarray = None, n_hard: int = 0) -> np.ndarray:
    n = llm_joined.shape[1]
    labeled = llm_joined._row_index.get(int(lid), set())
    labels = llm_joined._labels
    oracle_neg = np.array(
        [rid for rid in labeled if labels.get((int(lid), rid), -1) == 0],
        dtype=np.int64,
    )
    if len(oracle_neg) >= neg_k:
        return rng.choice(oracle_neg, size=neg_k, replace=False)
    parts = [oracle_neg]
    need = neg_k - len(oracle_neg)
    if hard_pool is not None and n_hard > 0:
        cand = np.array(
            [rid for rid in hard_pool if labels.get((int(lid), int(rid)), -1) != 1],
            dtype=np.int64,
        )
        if len(cand) > 0:
            take = min(n_hard, need, len(cand))
            parts.append(rng.choice(cand, size=take, replace=False))
            need -= take
    if need > 0:
        pseudo = rng.integers(0, n, size=max(need * 2, 8))
        pseudo = np.array([rid for rid in pseudo if rid not in labeled], dtype=np.int64)[:need]
        if len(pseudo) < need:
            extra = rng.integers(0, n, size=need - len(pseudo)).astype(np.int64)
            pseudo = np.concatenate([pseudo, extra])
        parts.append(pseudo)
    return np.concatenate(parts)


def _build_neg_matrix(lids, llm_joined, neg_k, rng, hard_pool=None, n_hard=0):
    return np.stack([
        _neg_rids(int(lid), llm_joined, neg_k, rng,
                  None if hard_pool is None else hard_pool[i], n_hard)
        for i, lid in enumerate(lids)
    ])


def _inbatch_mask(pairs: np.ndarray, pos_set: set) -> np.ndarray:
    lids, rids = pairs[:, 0], pairs[:, 1]
    mask = rids[None, :] != rids[:, None]
    pos_by_l = defaultdict(list)
    for lid, rid in pos_set:
        pos_by_l[int(lid)].append(int(rid))
    for lid, pos_rids in pos_by_l.items():
        rows = np.flatnonzero(lids == lid)
        if len(rows) == 0:
            continue
        mask[np.ix_(rows, np.flatnonzero(np.isin(rids, pos_rids)))] = False
    np.fill_diagonal(mask, False)
    return mask


def info_nce_loss(anchor, pos, neg, inbatch_mask, tau) -> torch.Tensor:
    s_pos = (anchor * pos).sum(-1, keepdim=True) / tau
    s_neg = torch.bmm(neg, anchor.unsqueeze(-1)).squeeze(-1) / tau
    s_ib = (anchor @ pos.t() / tau).masked_fill(~inbatch_mask, float("-inf"))
    logits = torch.cat([s_pos, s_neg, s_ib], dim=1)
    target = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, target)


@torch.no_grad()
def _val_mrr(model, x_all, r_all, anchor_fid, pos_rid, ban_mask) -> float:
    model.eval()
    scores = model(x_all[anchor_fid]) @ r_all.t()
    scores = scores.masked_fill(ban_mask, float("-inf"))
    pos_score = scores.gather(1, pos_rid.view(-1, 1))
    rank = (scores > pos_score).sum(1) + 1
    return float((1.0 / rank.float()).mean().item())


def _val_ban_mask(pairs: np.ndarray, pos_set: set, n: int) -> np.ndarray:
    mask = np.zeros((len(pairs), n), dtype=bool)
    pos_by_l = defaultdict(list)
    for lid, rid in pos_set:
        pos_by_l[int(lid)].append(int(rid))
    for i, (lid, rid) in enumerate(pairs):
        other = [r for r in pos_by_l.get(int(lid), []) if r != int(rid)]
        if other:
            mask[i, other] = True
    return mask


def train_projector(rtr, cfg: dict, device: torch.device, seed: int = 0):
    update_pair_feature_cache(rtr)
    llm_joined = rtr.llm_joined
    known_pos = llm_joined.get_positive_pairs()
    known_pos_set = {(int(lid), int(rid)) for lid, rid in known_pos}
    pos_pairs = np.asarray(
        [pair for pair in sorted(known_pos_set) if pair in rtr.feat_dict["pair_fid"]],
        dtype=np.int64,
    ).reshape(-1, 2)
    info = {
        "n_pos": len(pos_pairs), "trained": False, "epochs": 0,
        "train_loss": None, "val_score": None, "init_score": None,
        "best_epoch": 0, "n_train": 0, "n_val": 0,
        "n_mapped": len(rtr.feat_dict["pair_fid"]),
    }
    if len(pos_pairs) < cfg["dl_min_pos"]:
        info["reason"] = f"n_pos={len(pos_pairs)} < dl_min_pos={cfg['dl_min_pos']}"
        return None, info

    rng = np.random.default_rng(seed)
    anchor_fid = _anchor_fids(rtr, pos_pairs)
    n = len(pos_pairs)
    perm = rng.permutation(n)
    if n >= cfg["dl_val_min"]:
        n_val = max(1, int(n * cfg["dl_val_ratio"]))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
    else:
        val_idx, tr_idx = _EMPTY, perm

    feat_emb, remb = rtr.feat_emb, rtr.Remb
    dim = feat_emb.shape[1]
    x_all = torch.from_numpy(np.ascontiguousarray(feat_emb, dtype=np.float32)).to(device)
    r_all = torch.from_numpy(np.ascontiguousarray(remb, dtype=np.float32)).to(device)
    hidden = max(dim, int(dim * cfg["dl_hidden_ratio"]))
    torch.manual_seed(seed)
    model = MLPProjector(dim, hidden, cfg["dl_dropout"]).to(device).train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["dl_lr"], weight_decay=cfg["dl_weight_decay"])

    tau = cfg["dl_temperature"]
    neg_k = cfg["dl_neg_k"]
    batch_size = cfg["dl_batch_size"]
    n_hard = int(round(neg_k * cfg["dl_hard_ratio"]))
    hard_pool = _hard_neg_pool(rtr, anchor_fid, cfg["dl_hard_pool"]) if n_hard > 0 else None

    if len(val_idx) > 0:
        v_pairs = pos_pairs[val_idx]
        v_neg = _build_neg_matrix(
            v_pairs[:, 0], llm_joined, neg_k, np.random.default_rng(seed + 1),
            None if hard_pool is None else hard_pool[val_idx], n_hard)
        v_anchor = torch.from_numpy(anchor_fid[val_idx]).to(device)
        v_pos = torch.from_numpy(v_pairs[:, 1]).to(device)
        v_neg_t = torch.from_numpy(v_neg).to(device)
        v_mask = torch.from_numpy(_inbatch_mask(v_pairs, known_pos_set)).to(device)
        v_ban = torch.from_numpy(_val_ban_mask(v_pairs, known_pos_set, remb.shape[0])).to(device)
        init_score = _val_mrr(model, x_all, r_all, v_anchor, v_pos, v_ban)
        best_score = init_score
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        init_score = None
        best_score, best_state = -float("inf"), None
    bad_epochs, best_epoch, last_train_loss = 0, 0, None

    for epoch in range(cfg["dl_epochs"]):
        model.train()
        order = rng.permutation(len(tr_idx))
        total_loss, n_batch = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = tr_idx[order[start:start + batch_size]]
            if len(idx) < 2:
                continue
            batch_pairs = pos_pairs[idx]
            batch_neg = _build_neg_matrix(
                batch_pairs[:, 0], llm_joined, neg_k, rng,
                None if hard_pool is None else hard_pool[idx], n_hard)
            anchor = model(x_all[torch.from_numpy(anchor_fid[idx]).to(device)])
            pos = r_all[torch.from_numpy(batch_pairs[:, 1]).to(device)]
            neg = r_all[torch.from_numpy(batch_neg).to(device)]
            mask = torch.from_numpy(_inbatch_mask(batch_pairs, known_pos_set)).to(device)
            loss = info_nce_loss(anchor, pos, neg, mask, tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batch += 1
        if n_batch == 0:
            break
        last_train_loss = total_loss / n_batch
        info["epochs"] = epoch + 1
        if len(val_idx) == 0:
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            continue
        score = _val_mrr(model, x_all, r_all, v_anchor, v_pos, v_ban)
        if score > best_score + 1e-6:
            best_score, bad_epochs, best_epoch = score, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["dl_patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    info.update(
        trained=True, train_loss=last_train_loss, n_train=len(tr_idx), n_val=len(val_idx),
        best_epoch=best_epoch, val_score=None if best_score == -float("inf") else best_score,
        init_score=init_score,
    )
    if len(val_idx) > 0 and best_epoch == 0:
        info["trained"] = False
        info["reason"] = f"no epoch beat identity baseline (mrr={init_score:.4f})"
        return None, info
    return model.eval(), info


@torch.no_grad()
def _apply_projector(model, feat_emb, device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = np.empty_like(feat_emb, dtype=np.float32)
    for start in range(0, len(feat_emb), batch_size):
        chunk = torch.from_numpy(
            np.ascontiguousarray(feat_emb[start:start + batch_size], dtype=np.float32)).to(device)
        out[start:start + batch_size] = model(chunk).cpu().numpy()
    return np.ascontiguousarray(out, dtype=np.float32)


def _cap_drift(refined, original, max_drift: float) -> np.ndarray:
    delta = refined - original
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_drift / np.maximum(norm, 1e-12))
    capped = original + delta * scale
    capped = capped / np.maximum(np.linalg.norm(capped, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(capped, dtype=np.float32)


def dl_finetune(rtr, **args):
    cfg = _cfg(**args)
    device = torch.device(args.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(args.get("seed", 0)) + int(getattr(rtr, "iter_id", 0))
    model, info = train_projector(rtr, cfg, device, seed=seed)
    if model is None:
        print(f"[dl-finetune] feature={rtr.feature_type} skip ({info.get('reason', 'no model')})")
        x = rtr.feat_emb
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return np.ascontiguousarray(x, dtype=np.float32), info

    refined = _apply_projector(model, rtr.feat_emb, device)
    max_drift = float(cfg["dl_max_drift"])
    if max_drift > 0:
        refined = _cap_drift(refined, rtr.feat_emb, max_drift)
    drift = float(np.linalg.norm(refined - rtr.feat_emb, axis=1).mean())
    print(f"[dl-finetune] feature={rtr.feature_type} n_pos={info['n_pos']} "
          f"mapped={info['n_mapped']} epochs={info['epochs']} best@{info['best_epoch']} "
          f"val_mrr={None if info['val_score'] is None else round(info['val_score'], 4)} "
          f"init_mrr={None if info['init_score'] is None else round(info['init_score'], 4)} "
          f"emb_drift={drift:.4f}")
    return refined, info
