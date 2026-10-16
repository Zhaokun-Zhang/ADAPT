import hashlib
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ADAPT.paths import CACHE_DIR, ROOT
from ADAPT.utils import load_pickle, save_pickle


def _load_dotenv() -> dict:
    """Read Adapt-code/.env. Missing file or missing keys are left empty."""
    env: dict = {}
    for candidate in (ROOT / ".env", Path.cwd() / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
        break
    return env


def _is_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "modules.json").is_file()


def _embedding_model_path(embedding_model_name: str) -> str:
    """Use EMBEDDING_MODEL_DIR from .env, otherwise the Hugging Face model id."""
    folder = {
        "e5": "e5-base-v2",
        "mpnet": "all-mpnet-base-v2",
    }.get(embedding_model_name, embedding_model_name)
    configured = _load_dotenv().get("EMBEDDING_MODEL_DIR", "").strip()
    if not configured:
        if embedding_model_name == "e5":
            return "intfloat/e5-base-v2"
        return embedding_model_name
    base = Path(configured).expanduser()
    if _is_checkpoint(base):
        return str(base)
    nested = base / folder
    if nested.is_dir():
        return str(nested)
    raise FileNotFoundError(
        f"EMBEDDING_MODEL_DIR={configured} is not a checkpoint and does not "
        f"contain {folder}/. Set it in Adapt-code/.env, or leave it unset to "
        "download the Hugging Face model."
    )


def load_sentence_transformer(embedding_model_name: str = "e5", device: str = "cpu", **args):
    model_path = _embedding_model_path(embedding_model_name)
    print(f"Load embedding model {model_path}, device:{device}")
    return SentenceTransformer(model_path, device=device)


def _dict_hash(dic: dict) -> str:
    data_byte = pickle.dumps(sorted(dic.items()))
    return hashlib.md5(data_byte).hexdigest()


class EmbeddingModel:
    def __init__(self, embedding_model_name: str = "e5", device: str = "cpu", **args):
        self.cache_dir = CACHE_DIR / "embedding"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = load_sentence_transformer(
            embedding_model_name=embedding_model_name, device=device)

    def encode(self, texts, normalize_embeddings: bool = True, **args):
        params = {
            "sentences": list(texts),
            "normalize_embeddings": normalize_embeddings,
            "batch_size": 128,
            "show_progress_bar": True,
        }
        return self._cached_encode(params)

    def _cached_encode(self, params):
        hash_code = _dict_hash(params)
        path = self.cache_dir / hash_code
        if path.exists():
            print("  Embedding cache hit.")
            return load_pickle(str(path))
        emb_arr = np.asarray(self.model.encode(**params))
        save_pickle(emb_arr, str(path))
        return emb_arr
