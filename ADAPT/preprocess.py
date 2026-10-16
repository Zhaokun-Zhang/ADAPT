"""Driving-side selection, four-channel feature extraction, and text caches."""

import copy
import json
import os

import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from ADAPT.candidate import CandidateSet
from ADAPT.dataset import GlobalInfo, JoinDataset
from ADAPT.paths import CACHE_DIR, FEATURE_TYPES, PROMPT_DIR
from ADAPT.utils import load_json, save_json
from model import EmbeddingModel, LLM


def _fill_template(template_name: str, replace_dict: dict):
    prompt = load_json(str(PROMPT_DIR / template_name))
    for msg in prompt:
        content = msg["content"]
        if isinstance(content, list):
            content = "".join(content)
        for key, val in replace_dict.items():
            content = content.replace(key, "" if val is None else str(val))
        msg["content"] = content
    return prompt


def _call_llm_task(llm: LLM, prompt: list, task: str, max_retry: int = 3):
    call_info = {"Task": task, "Prompt": prompt}
    for attempt in range(max_retry):
        result = llm([call_info], enable_thinking=True)[0]
        if result is not None:
            return result
        print(f"[data_profiling] task {task} parse failed, retry {attempt + 1}/{max_retry}")
    raise RuntimeError(f"[data_profiling] task {task} failed after {max_retry} retries.")


def _sample_tuples(data: JoinDataset, **args):
    k = args.get("profiling_sample_k", 10)
    rng = np.random.default_rng(args.get("seed", 42))
    left, right = data.L.values[:, 1], data.R.values[:, 1]
    lidx = rng.choice(len(left), size=min(k, len(left)), replace=False)
    ridx = rng.choice(len(right), size=min(k, len(right)), replace=False)
    return [str(left[i]) for i in lidx], [str(right[i]) for i in ridx]


def _generate_data_description(data, llm, left_samples, right_samples):
    prompt = _fill_template("data_description.template", {
        "#PREDICATE#": data.predicate,
        "#L_SAMPLES#": json.dumps(left_samples, ensure_ascii=False, indent=2),
        "#R_SAMPLES#": json.dumps(right_samples, ensure_ascii=False, indent=2),
    })
    result = _call_llm_task(llm, prompt, task="data_description")
    data.Ldescription = result["L_description"]
    data.Rdescription = result["R_description"]
    print(f"[data_profiling] L_description: {data.Ldescription}")
    print(f"[data_profiling] R_description: {data.Rdescription}")


def _determine_driving_table(data, llm, left_samples, right_samples) -> bool:
    prompt = _fill_template("driving_or_driven.template", {
        "#PREDICATE#": data.predicate,
        "#L_DESCRIPTION#": data.Ldescription,
        "#R_DESCRIPTION#": data.Rdescription,
        "#L_SIZE#": str(len(data.L)),
        "#R_SIZE#": str(len(data.R)),
        "#L_SAMPLES#": json.dumps(left_samples, ensure_ascii=False, indent=2),
        "#R_SAMPLES#": json.dumps(right_samples, ensure_ascii=False, indent=2),
    })
    result = _call_llm_task(llm, prompt, task="driving_or_driven")
    print(f"[data_profiling] driving_table = {result['driving_table']} "
          f"(rule_applied={result.get('rule_applied')}); reason: {result.get('reason')}")
    return result["driving_table"] == "L"


def _swap_driving_driven(data: JoinDataset):
    data.Lname, data.Rname = data.Rname, data.Lname
    data.L, data.R = data.R, data.L
    data.L = data.L.reset_index(drop=True)
    data.R = data.R.reset_index(drop=True)
    data.L.columns = ["Lid", data.Lname]
    data.R.columns = ["Rid", data.Rname]
    data.Ldescription, data.Rdescription = data.Rdescription, data.Ldescription
    data.predicate = (data.predicate
                      .replace("{left_tuple}", "{#RIGHT_TUPLE#}")
                      .replace("{right_tuple}", "{left_tuple}")
                      .replace("{#RIGHT_TUPLE#}", "{right_tuple}"))
    data.joined = data.joined[["Rid", "Lid"]].copy()
    data.joined.columns = ["Lid", "Rid"]
    print("[data_profiling] swapped L and R so the driving table is L.")
    print(f"[data_profiling] predicate: {data.predicate}")


def _profiling_cache_path(data: JoinDataset) -> str:
    return str(CACHE_DIR / data.Dname / "profiling.json")


def data_profiling(data: JoinDataset, llm: LLM, **args):
    cache_path = _profiling_cache_path(data)
    if not os.path.exists(cache_path):
        left_samples, right_samples = _sample_tuples(data, **args)
        _generate_data_description(data, llm, left_samples, right_samples)
        left_is_driving = _determine_driving_table(data, llm, left_samples, right_samples)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        save_json({
            "L_description": data.Ldescription,
            "R_description": data.Rdescription,
            "L_is_driving": left_is_driving,
            "LLM_usage": llm.get_usage(),
        }, cache_path, indent=2)
    else:
        print(f"[data_profiling] cache hit: {cache_path}")
        cached = load_json(cache_path)
        data.Ldescription = cached["L_description"]
        data.Rdescription = cached["R_description"]
        left_is_driving = cached["L_is_driving"]
        usage = cached["LLM_usage"]
        llm._update_usage(usage.get("data_description", {}), "data_description")
        llm._update_usage(usage.get("driving_or_driven", {}), "driving_or_driven")
    if not left_is_driving:
        _swap_driving_driven(data)


def _chunkize(texts):
    doc_list = []
    for text in tqdm(texts, desc="sentence chunking"):
        doc_list.append(sent_tokenize(str(text), language="english"))
    return doc_list


def _tfidf(docs, **args):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, args.get("tfidf_word_len", 1)),
        max_df=args.get("tfidf_max_df", 0.9),
        min_df=args.get("tfidf_min_df", 3),
        max_features=args.get("tfidf_word_num", 10000),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]*\b",
    )
    matrix = vectorizer.fit_transform(docs)
    return matrix, vectorizer.get_feature_names_out()


def _word_documents(texts, emb_model, remb, **args):
    threshold = args.get("kw_thred", 0.9)
    matrix, keywords = _tfidf(texts, **args)
    keyword_emb = emb_model.encode(keywords, normalize_embeddings=True)
    keyword_score = np.max(keyword_emb @ remb.T, axis=1)
    kept = np.where(keyword_score > threshold)[0]
    minimum = max(1, len(texts) // 3)
    if len(kept) < minimum:
        below = np.where(keyword_score <= threshold)[0]
        supplement = below[np.argsort(-keyword_score[below])][:minimum - len(kept)]
        kept = np.concatenate([kept, supplement])
    kept_words = set(keywords[kept])
    documents = []
    for docid in range(len(texts)):
        present = set(keywords[matrix[docid].toarray().flatten() > 0]) & kept_words
        documents.append(list(present))
    return documents


def _map_prompts(data: JoinDataset, texts):
    template = load_json(str(PROMPT_DIR / "map_feature_extraction.template"))
    calls = []
    for doc in texts:
        prompt = copy.deepcopy(template)
        prompt[1]["content"] = (prompt[1]["content"]
                                .replace("#R_DESCRIPTION#", data.Rdescription or "")
                                .replace("#L_DESCRIPTION#", data.Ldescription or "")
                                .replace("#LEFT_TUPLE#", str(doc))
                                .replace("#PREDICATE#", data.predicate))
        calls.append({"Task": "map", "Prompt": prompt})
    return calls


def _feature_cache_path(data: JoinDataset) -> str:
    return str(CACHE_DIR / data.Dname / "features.json")


def _embed_channels(data: JoinDataset, emb_model: EmbeddingModel, channels: dict):
    print("Driven-side embedding...")
    data.Remb = np.ascontiguousarray(
        emb_model.encode(data.R.values[:, 1], normalize_embeddings=True), dtype=np.float32)
    for feature_type in FEATURE_TYPES:
        data.features[feature_type] = data.build_feature_index(channels[feature_type], emb_model)


def feature_extraction(data: JoinDataset, emb_model: EmbeddingModel, llm: LLM, **args):
    cache_path = _feature_cache_path(data)
    if os.path.exists(cache_path):
        print(f"[features] cache hit: {cache_path}")
        cached = load_json(cache_path)
        channels = {feature_type: cached[feature_type] for feature_type in FEATURE_TYPES}
        _embed_channels(data, emb_model, channels)
        llm._update_usage(cached.get("map_cost", {}), "map")
        return

    texts = data.L.values[:, 1]
    print("Driven-side embedding...")
    data.Remb = np.ascontiguousarray(
        emb_model.encode(data.R.values[:, 1], normalize_embeddings=True), dtype=np.float32)

    print("Document channel...")
    doc_list = [[str(doc)] for doc in texts]
    data.features["doc"] = data.build_feature_index(doc_list, emb_model)

    print("Sentence channel...")
    data.features["sent"] = data.build_feature_index(_chunkize(texts), emb_model)

    print("Word channel...")
    word_docs = _word_documents(texts, emb_model, data.Remb, **args)
    data.features["word"] = data.build_feature_index(word_docs, emb_model)

    print("LLM extraction channel...")
    calls = _map_prompts(data, texts)
    extracted = llm(calls, enable_thinking=False)
    ext_docs = []
    for features, call in zip(extracted, calls):
        while features is None or not isinstance(features, list):
            features = llm([call], enable_thinking=False)[0]
        ext_docs.append([str(item) for item in features if str(item).strip()])
    data.features["ext"] = data.build_feature_index(ext_docs, emb_model)

    saved = {feature_type: data.features[feature_type]["doc_list"] for feature_type in FEATURE_TYPES}
    saved["map_cost"] = llm.get_usage().get("map", {})
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_json(saved, cache_path, indent=2)
    print(f"[features] wrote {cache_path}")


def data_preprocess(data: JoinDataset, **args) -> GlobalInfo:
    embedding_model = EmbeddingModel(
        embedding_model_name=args.get("embedding_model_name", "e5"),
        device=args.get("device", "cpu"),
    )
    llm = LLM(
        max_workers=args.get("llm_max_workers", 16),
        mode=args.get("llm_mode", "cache"),
    )
    data_profiling(data, llm, **args)
    llm.build_llm_joined_cache(data)
    if args.get("llm_mode") == "simulation":
        llm.build_simulation_joined(data.joined.values)
    feature_extraction(data, embedding_model, llm, **args)
    return GlobalInfo(
        data=data,
        emb_model=embedding_model,
        llm=llm,
        all_pos_cand=CandidateSet(feature_type_list=list(FEATURE_TYPES)),
        **args,
    )
