"""Dataset loading and the mutable state shared across join iterations."""

import os
import sys
from collections import defaultdict
from typing import List

import faiss
import numpy as np
import pandas as pd

from ADAPT.paths import DATA_DIR, RESULT_DIR
from ADAPT.sparse_label_matrix import SparseLabelMatrix
from ADAPT.utils import load_text, save_json


class Recorder:
    def __init__(self, **args):
        self.args = args
        dataset = args.get("Dataset", args.get("dataset", "run"))
        mark = args.get("mark", "default")
        self.log_dir = str(RESULT_DIR / "logs" / str(dataset) / str(mark))
        self.data: dict = defaultdict(list)
        self.float_digit = 4
        self.key_metrics: List[str] = [
            "cmp_num", "proxy_TP", "proxy_recall",
            "e2e_TP", "e2e_recall", "e2e_precision",
        ]
        os.makedirs(self.log_dir, exist_ok=True)
        self.save_args()
        if args.get("redirect_log", False):
            self.redirect_std()

    def redirect_std(self):
        log_file = open(f"{self.log_dir}/running.log", "w", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file

    def add(self, key, val):
        if isinstance(val, float):
            val = round(val, self.float_digit)
        self.data[key].append(val)

    def save(self):
        save_path = f"{self.log_dir}/record.csv"
        try:
            frame = pd.DataFrame(dict(self.data))
            frame.index.name = "iter_id"
            left = [col for col in self.key_metrics if col in frame.columns]
            other = [col for col in frame.columns if col not in self.key_metrics]
            frame[left + other].to_csv(save_path, index=True)
        except Exception:
            save_json(dict(self.data), save_path, indent=2)

    def save_args(self):
        save_json(self.args, f"{self.log_dir}/args.json", indent=2)


class GlobalInfo:
    def __init__(self, data, emb_model, llm, all_pos_cand, **args):
        self.dataset = data
        self.embedding_model = emb_model
        self.llm = llm
        self.iter_id = 0
        self.pos_cand = None
        self.all_pos_cand = all_pos_cand
        self.all_pos_cand_list: list = []
        self.total_quota = int(args["per_round_budget"])
        self.early_stop = False
        self.pull_log: list = []
        self.pull_logged_pairs: set = set()
        self.audit_log: list = []
        self.last_recall_certificate = None
        self.recall_certificate_history: list = []
        self.anytime_candidate_pool = set()
        self.anytime_near_pool = set()
        self.llm_joined = SparseLabelMatrix(data.L.shape[0], data.R.shape[0])
        self.recorder = Recorder(**data.get_info(), **args)
        self.dim = data.Remb.shape[1]
        self.Rindex = faiss.IndexFlatIP(self.dim)
        self.Rindex.add(np.ascontiguousarray(data.Remb, dtype=np.float32))

    def evaluate(self):
        llm_joined_pairs = self.llm_joined.get_positive_pairs()
        labeled_pairs = self.llm_joined.get_labeled_pairs()
        joined = self.dataset.joined.values.copy()
        dtype = np.dtype([("", "int"), ("", "int")])

        def _sets(pred):
            pred_set = set(pred.view(dtype=dtype).reshape(-1).tolist()) if len(pred) else set()
            actual_set = set(joined.view(dtype=dtype).reshape(-1).tolist()) if len(joined) else set()
            tp = len(pred_set & actual_set)
            fp = len(pred_set - actual_set)
            fn = len(actual_set - pred_set)
            recall = tp / (tp + fn) if (tp + fn) else 0
            precision = tp / (tp + fp) if (tp + fp) else 0
            return tp, fp, fn, recall, precision

        print("**************************")
        print("* End To End Evaluation  *")
        tp, fp, fn, recall, precision = _sets(llm_joined_pairs)
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Recall: {recall}")
        print(f"  Precision: {precision}")
        effectiveness = {
            "e2e_recall": recall, "e2e_precision": precision,
            "e2e_TP": tp, "e2e_FP": fp, "e2e_FN": fn,
        }

        print("* Proxy Model Evaluation *")
        tp, fp, fn, recall, precision = _sets(labeled_pairs)
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Recall: {recall}")
        print(f"  Precision: {precision}")
        proxy = {
            "proxy_recall": recall, "proxy_precision": precision,
            "proxy_TP": tp, "proxy_FP": fp, "proxy_FN": fn,
        }
        print("*        LLM Usage        *")
        self.llm.prt_usage()

        self.recorder.add("cmp_num", len(labeled_pairs))
        for key, val in effectiveness.items():
            self.recorder.add(key, val)
        for key, val in proxy.items():
            self.recorder.add(key, val)
        for cost_type, usage in self.llm.get_usage().items():
            if cost_type == "current_batch":
                continue
            for key, val in usage.items():
                self.recorder.add(f"{cost_type}_{key}", val)
        return {"effectiveness": effectiveness, "proxy": proxy, "efficiency": self.llm.get_usage()}


class JoinDataset:
    def __init__(self, dataset: str, **args):
        self.data_dir = args.get("data_dir", DATA_DIR)
        self.Dname = dataset
        self.Lname = None
        self.Ldescription = None
        self.L = None
        self.Rname = None
        self.Rdescription = None
        self.R = None
        self.predicate = None
        self.detailed_predicate = None
        self.joined = None
        self.Remb = None
        self.features = {}
        self._load(dataset)

    def _load(self, dataset: str):
        folder = os.path.join(self.data_dir, dataset)
        required = ["L.parquet", "R.parquet", "joined.csv", "predicate.txt"]
        missing = [name for name in required if not os.path.exists(os.path.join(folder, name))]
        if missing:
            raise FileNotFoundError(
                f"Dataset '{dataset}' is missing {missing} under {folder}. "
                "Each dataset directory needs L.parquet, R.parquet, joined.csv, and predicate.txt."
            )
        self.L = pd.read_parquet(os.path.join(folder, "L.parquet"))
        self.R = pd.read_parquet(os.path.join(folder, "R.parquet"))
        self.joined = pd.read_csv(os.path.join(folder, "joined.csv"))
        self.predicate = load_text(os.path.join(folder, "predicate.txt")).strip()
        self.Lname = str(self.L.columns[1])
        self.Rname = str(self.R.columns[1])
        self.L.columns = ["Lid", self.Lname]
        self.R.columns = ["Rid", self.Rname]
        self.joined.columns = ["Lid", "Rid"]
        self.L = self.L.reset_index(drop=True)
        self.R = self.R.reset_index(drop=True)
        self.print_info()

    def print_info(self):
        print("Dataset:", self.Dname)
        print("predicate:", self.predicate)
        print("L num:", len(self.L))
        print("R num:", len(self.R))
        print("Joined Num:", len(self.joined))
        print("Cmp Num:", len(self.L) * len(self.R))
        selectivity = len(self.joined) / len(self.L) / len(self.R)
        print("select ratio:", f"{selectivity * 100:.4f}", "%")

    def get_info(self):
        return {
            "Dataset": self.Dname,
            "Predicate": self.predicate,
            "L num": len(self.L),
            "R num": len(self.R),
            "Joined Num": len(self.joined),
            "Cmp Num": len(self.L) * len(self.R),
            "Select Ratio": len(self.joined) / len(self.L) / len(self.R),
        }

    def build_feature_index(self, doclist, embedding_model):
        sent2docids = defaultdict(list)
        for docid, doc in enumerate(doclist):
            for sent in doc:
                sent2docids[sent].append(docid)
        feat_list = list(sent2docids.keys())
        fid2docids = {}
        docid2fids = defaultdict(list)
        for sid, sent in enumerate(feat_list):
            fid2docids[sid] = sent2docids[sent]
            for docid in sent2docids[sent]:
                docid2fids[docid].append(sid)
        embeddings = embedding_model.encode(feat_list, normalize_embeddings=True)
        return {
            "doc_list": doclist,
            "feat_list": feat_list,
            "feat_emb": np.ascontiguousarray(embeddings, dtype=np.float32),
            "fid2docids": fid2docids,
            "docid2fids": dict(docid2fids),
            "pair_fid": {},
        }
