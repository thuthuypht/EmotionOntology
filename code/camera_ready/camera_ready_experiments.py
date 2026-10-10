#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera-ready experiments for the CSoNet 2026 paper:
"Ontology-Driven Sentiment Analytics for Teaching Outcome Assessment from
Vietnamese Open-Ended Student Feedback"

Goals
-----
1. Reproduce the lightweight control / topic-metadata / ontology-assisted protocol.
2. Add a strong Vietnamese pretrained-language-model baseline using PhoBERT-v2.
3. Run five seeds on exactly the same synthetic_train.csv / synthetic_val.csv split.
4. Export per-seed metrics, mean +/- std, a paper-ready table, dataset statistics,
   ontology/KG statistics, IAA summary, representative SPARQL results, and
   camera-ready replacement text.

The script is designed for Kaggle/Colab GPU but can run the lightweight part on CPU.
It resumes at seed granularity: completed seed JSON files are reused unless --force.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import unicodedata
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

# Optional heavy imports are delayed until PhoBERT is requested.

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = ["negative", "neutral", "positive"]

# Six-dimensional topic space used by the current manuscript.
TOPIC_LIST = [
    "Facilities",
    "TeachingMethod",
    "CourseContent",
    "Material",
    "LecturerAttitude",
    "OtherTopic",
]
TOPIC_MAP_BENCH = {
    "facility": "Facilities",
    "curriculum": "CourseContent",
    "lecturer": "TeachingMethod",
    "others": "OtherTopic",
}

# Curated lexical cues grounded in the manuscript and the supplied NTU annotation rules.
CURATED_TOPIC = {
    "Facilities": [
        "cơ sở vật chất", "phòng học", "phòng máy", "máy chiếu", "wifi", "mạng",
        "thiết bị", "điều hòa", "máy tính", "âm thanh", "micro", "bàn ghế",
        "facility", "facilities", "classroom", "lab", "equipment", "projector",
    ],
    "TeachingMethod": [
        "phương pháp giảng dạy", "cách dạy", "giảng dạy", "truyền đạt kiến thức",
        "giảng bài", "dạy học", "thực hành", "ví dụ thực tế", "dạy nhanh", "dạy chậm",
        "teaching method", "lecture pace", "teaching", "practice", "interaction",
    ],
    "CourseContent": [
        "nội dung môn học", "chương trình", "chương trình học", "đề cương", "kiến thức",
        "chủ đề", "nội dung môn", "curriculum", "course content", "syllabus", "topics",
    ],
    "Material": [
        "tài liệu", "giáo trình", "slide", "bài giảng", "tài liệu học", "lms",
        "e-learning", "sách giáo khoa", "materials", "textbook", "slides", "lecture notes",
    ],
    "LecturerAttitude": [
        "thái độ", "thái độ giảng viên", "nhiệt tình", "tận tâm", "thân thiện", "hỗ trợ",
        "giải đáp", "tôn trọng sinh viên", "giao tiếp với sinh viên", "lecturer attitude",
        "supportive", "helpful", "friendly", "responsive",
    ],
    "OtherTopic": ["khác", "không có ý kiến", "không ý kiến", "other", "no comment"],
}

CURATED_SENTIMENT = {
    "Negative": [
        "khó hiểu", "kém", "tệ", "mơ hồ", "không rõ", "không hài lòng", "quá nhanh",
        "quá chậm", "thiếu", "không hay", "không tốt", "không hiểu", "chưa tốt",
        "dạy chán", "nói nhỏ", "lòng vòng", "không ổn", "dạy qua loa", "thất vọng",
        "không thể chấp nhận", "vô trách nhiệm", "bad", "poor", "confusing", "unclear",
        "not satisfied", "too fast", "too slow", "lack",
    ],
    "Neutral": [
        "bình thường", "tạm", "khá", "không ý kiến", "không có ý kiến", "mong", "đề nghị",
        "hy vọng", "bổ sung", "cải thiện", "neutral", "average", "okay", "no opinion",
    ],
    "Positive": [
        "tốt", "tuyệt vời", "hữu ích", "rõ ràng", "dễ hiểu", "hiệu quả", "hài lòng",
        "xuất sắc", "rất hay", "giỏi", "nhiệt tình", "tận tâm", "thân thiện", "chu đáo",
        "có tâm", "good", "great", "excellent", "helpful", "clear", "effective", "satisfied",
    ],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def normalize_text(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_basic(text: object) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ỹđĐ]+", normalize_text(text), flags=re.UNICODE)


def ngrams(tokens: Sequence[str], n: int) -> List[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def metric_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "Accuracy": float(acc),
        "Macro-Precision": float(p),
        "Macro-Recall": float(r),
        "Macro-F1": float(f1),
    }


def load_synthetic(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = input_dir / "synthetic_train.csv"
    val_path = input_dir / "synthetic_val.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path.name} and {val_path.name} in {input_dir}"
        )
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    required = {"sentence", "sentiment", "topic"}
    for name, df in [("train", train), ("val", val)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        df["sentence"] = df["sentence"].map(normalize_text)
        df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
        df["topic"] = df["topic"].astype(str).str.lower().str.strip()
        if not set(df["sentiment"]).issubset(LABEL_MAP):
            raise ValueError(f"Unexpected sentiment labels in {name}: {sorted(set(df['sentiment']))}")
        if not set(df["topic"]).issubset(TOPIC_MAP_BENCH):
            raise ValueError(f"Unexpected topic labels in {name}: {sorted(set(df['topic']))}")
        df["label"] = df["sentiment"].map(LABEL_MAP).astype(int)
        df["topic_owl"] = df["topic"].map(TOPIC_MAP_BENCH)
    return train, val


def export_dataset_statistics(train: pd.DataFrame, val: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for split, df in [("synthetic_train", train), ("synthetic_val", val)]:
        rows.append({"split": split, "dimension": "TOTAL", "category": "all", "count": len(df)})
        for label in ["negative", "neutral", "positive"]:
            rows.append({
                "split": split,
                "dimension": "sentiment",
                "category": label,
                "count": int((df["sentiment"] == label).sum()),
            })
        for topic in ["curriculum", "facility", "lecturer", "others"]:
            rows.append({
                "split": split,
                "dimension": "topic",
                "category": topic,
                "count": int((df["topic"] == topic).sum()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "dataset_statistics.csv", index=False, encoding="utf-8-sig")
    return out


class OntologyLexicalFeatureBuilder:
    """Reconstructs the manuscript's 33-D ontology lexical feature protocol.

    Blocks:
      6 topic lexical match counts
      3 sentiment lexical match counts
      6 benchmark-topic metadata one-hot values
      18 topic x sentiment interactions
    Total = 33 dimensions.

    Additional discriminative lexical cues are learned only from the seed-specific
    training portion; the fixed final synthetic_val split is never used to build cues.
    """

    def __init__(self, top_sent: int = 160, min_sent: int = 10, top_topic: int = 80, min_topic: int = 12):
        self.top_sent = top_sent
        self.min_sent = min_sent
        self.top_topic = top_topic
        self.min_topic = min_topic
        self.topic_labels: Dict[str, set] = {k: set(map(normalize_text, v)) for k, v in CURATED_TOPIC.items()}
        self.sent_labels: Dict[str, set] = {k: set(map(normalize_text, v)) for k, v in CURATED_SENTIMENT.items()}
        self.topic_ng = {}
        self.sent_ng = {}

    @staticmethod
    def _label_ngram_set(labels: Iterable[str], max_n: int = 3) -> Dict[int, set]:
        sets = {1: set(), 2: set(), 3: set()}
        for lab in labels:
            toks = tokenize_basic(lab)
            for n in range(1, min(max_n, len(toks)) + 1):
                for ng in ngrams(toks, n):
                    if len(ng) >= 3:
                        sets[n].add(ng)
        return sets

    @staticmethod
    def _text_ngram_sets(text: str) -> Dict[int, set]:
        toks = tokenize_basic(text)
        return {1: set(ngrams(toks, 1)), 2: set(ngrams(toks, 2)), 3: set(ngrams(toks, 3))}

    @staticmethod
    def _count_matches(text_sets: Dict[int, set], label_sets: Dict[int, set]) -> float:
        return float(sum(len(text_sets[n].intersection(label_sets[n])) for n in [1, 2, 3]))

    def fit(self, train_df: pd.DataFrame) -> "OntologyLexicalFeatureBuilder":
        # Sentiment-discriminative unigram/bigram cues using training labels only.
        class_counts = {0: Counter(), 1: Counter(), 2: Counter()}
        total_counts = Counter()
        for text, y in zip(train_df["sentence"], train_df["label"]):
            toks = tokenize_basic(text)
            feats = toks + ngrams(toks, 2)
            for f in feats:
                if len(f) < 3:
                    continue
                class_counts[int(y)][f] += 1
                total_counts[f] += 1

        def top_log_odds(target_cls: int) -> List[str]:
            scores = []
            sum_t = sum(class_counts[target_cls].values()) + 1.0
            sum_all = sum(total_counts.values()) + 1.0
            sum_rest = (sum_all - sum_t) + 1.0
            for term, c_t in class_counts[target_cls].items():
                if total_counts[term] < self.min_sent:
                    continue
                c_r = total_counts[term] - c_t
                p_t = (c_t + 1.0) / sum_t
                p_r = (c_r + 1.0) / sum_rest
                scores.append((math.log(p_t / p_r), term))
            scores.sort(reverse=True)
            return [term for _, term in scores[: self.top_sent]]

        self.sent_labels["Negative"].update(top_log_odds(0))
        self.sent_labels["Neutral"].update(top_log_odds(1))
        self.sent_labels["Positive"].update(top_log_odds(2))

        # Frequent topic phrases from training only.
        topic_counts = {t: Counter() for t in TOPIC_LIST}
        for text, tp in zip(train_df["sentence"], train_df["topic_owl"]):
            toks = tokenize_basic(text)
            feats = toks + ngrams(toks, 2)
            for f in feats:
                if len(f) >= 3:
                    topic_counts[tp][f] += 1

        for tp in ["Facilities", "TeachingMethod", "CourseContent", "OtherTopic"]:
            items = [(c, term) for term, c in topic_counts[tp].items() if c >= self.min_topic]
            items.sort(reverse=True)
            self.topic_labels[tp].update(term for _, term in items[: self.top_topic])

        self.topic_ng = {tp: self._label_ngram_set(self.topic_labels[tp]) for tp in TOPIC_LIST}
        self.sent_ng = {s: self._label_ngram_set(self.sent_labels[s]) for s in ["Negative", "Neutral", "Positive"]}
        return self

    def topic_metadata(self, df: pd.DataFrame) -> np.ndarray:
        tp_to_idx = {t: i for i, t in enumerate(TOPIC_LIST)}
        out = np.zeros((len(df), len(TOPIC_LIST)), dtype=np.float32)
        for i, tp in enumerate(df["topic_owl"]):
            out[i, tp_to_idx.get(tp, tp_to_idx["OtherTopic"])] = 1.0
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        topic_mc = np.zeros((len(df), 6), dtype=np.float32)
        sent_mc = np.zeros((len(df), 3), dtype=np.float32)
        for i, text in enumerate(df["sentence"]):
            text_sets = self._text_ngram_sets(text)
            for j, tp in enumerate(TOPIC_LIST):
                topic_mc[i, j] = self._count_matches(text_sets, self.topic_ng[tp])
            for j, s in enumerate(["Negative", "Neutral", "Positive"]):
                sent_mc[i, j] = self._count_matches(text_sets, self.sent_ng[s])
        meta = self.topic_metadata(df)
        cross = np.einsum("ij,ik->ijk", meta, sent_mc).reshape(len(df), -1).astype(np.float32)
        return np.hstack([topic_mc, sent_mc, meta, cross]).astype(np.float32)


@dataclass
class MetaResult:
    pred: np.ndarray
    c_value: float
    threshold: float
    override_count: int


def fit_conservative_meta(
    dev_logits: np.ndarray,
    dev_extra: np.ndarray,
    dev_y: np.ndarray,
    test_logits: np.ndarray,
    test_extra: np.ndarray,
    seed: int,
) -> MetaResult:
    """Train the conservative second-stage meta-classifier without using final test labels."""
    idx = np.arange(len(dev_y))
    dtr, dva = train_test_split(idx, test_size=0.4, random_state=seed, stratify=dev_y)
    base_dev_pred = dev_logits.argmax(axis=1)
    base_sel = metric_dict(dev_y[dva], base_dev_pred[dva])

    best = None
    best_score = -1e9
    for c_value in [0.02, 0.05, 0.1, 0.2, 0.5]:
        clf = LogisticRegression(max_iter=700, C=c_value, solver="lbfgs")
        clf.fit(np.hstack([dev_logits[dtr], dev_extra[dtr]]), dev_y[dtr])
        probs = clf.predict_proba(np.hstack([dev_logits[dva], dev_extra[dva]]))
        meta_pred = probs.argmax(axis=1)
        meta_conf = probs.max(axis=1)
        for thr in [0.55, 0.60, 0.65, 0.70, 0.75]:
            pred = base_dev_pred[dva].copy()
            mask = (meta_conf >= thr) & (meta_pred != pred)
            pred[mask] = meta_pred[mask]
            m = metric_dict(dev_y[dva], pred)
            if m["Accuracy"] > base_sel["Accuracy"] and m["Macro-F1"] > base_sel["Macro-F1"]:
                score = (m["Accuracy"] - base_sel["Accuracy"]) + (m["Macro-F1"] - base_sel["Macro-F1"])
                if score > best_score:
                    best_score = score
                    best = (c_value, thr)

    if best is None:
        best = (0.1, 0.80)

    c_value, thr = best
    clf = LogisticRegression(max_iter=700, C=c_value, solver="lbfgs")
    clf.fit(np.hstack([dev_logits, dev_extra]), dev_y)
    probs = clf.predict_proba(np.hstack([test_logits, test_extra]))
    meta_pred = probs.argmax(axis=1)
    meta_conf = probs.max(axis=1)
    base_test_pred = test_logits.argmax(axis=1)
    final_pred = base_test_pred.copy()
    mask = (meta_conf >= thr) & (meta_pred != base_test_pred)
    final_pred[mask] = meta_pred[mask]
    return MetaResult(final_pred, c_value, thr, int(mask.sum()))


def run_lightweight_seed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    seed_dir: Path,
) -> List[Dict[str, object]]:
    """Fast CPU replication of the manuscript control and ontology-assisted variants."""
    idx = np.arange(len(train_df))
    tr_idx, dev_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=train_df["label"]
    )
    tr = train_df.iloc[tr_idx].reset_index(drop=True)
    dev = train_df.iloc[dev_idx].reset_index(drop=True)
    test = test_df.reset_index(drop=True)

    hv_word = HashingVectorizer(
        analyzer="word", ngram_range=(1, 2), n_features=2**16,
        alternate_sign=False, norm=None, lowercase=True,
    )
    hv_char = HashingVectorizer(
        analyzer="char", ngram_range=(3, 5), n_features=2**16,
        alternate_sign=False, norm=None, lowercase=True,
    )
    Xtr_raw = hstack([hv_word.transform(tr["sentence"]), hv_char.transform(tr["sentence"])], format="csr")
    Xdev_raw = hstack([hv_word.transform(dev["sentence"]), hv_char.transform(dev["sentence"])], format="csr")
    Xtest_raw = hstack([hv_word.transform(test["sentence"]), hv_char.transform(test["sentence"])], format="csr")
    tfidf = TfidfTransformer()
    Xtr = tfidf.fit_transform(Xtr_raw)
    Xdev = tfidf.transform(Xdev_raw)
    Xtest = tfidf.transform(Xtest_raw)

    clf = SGDClassifier(
        loss="log_loss", alpha=1e-5, max_iter=1500, tol=1e-3,
        shuffle=True, random_state=seed, class_weight="balanced", n_jobs=1,
    )
    clf.fit(Xtr, tr["label"].to_numpy())
    dev_logits = clf.decision_function(Xdev)
    test_logits = clf.decision_function(Xtest)
    base_pred = test_logits.argmax(axis=1)

    builder = OntologyLexicalFeatureBuilder().fit(tr)
    dev_topic = builder.topic_metadata(dev)
    test_topic = builder.topic_metadata(test)
    dev_onto = builder.transform(dev)
    test_onto = builder.transform(test)

    topic_res = fit_conservative_meta(dev_logits, dev_topic, dev["label"].to_numpy(), test_logits, test_topic, seed)
    onto_res = fit_conservative_meta(dev_logits, dev_onto, dev["label"].to_numpy(), test_logits, test_onto, seed)

    rows = []
    for model_name, pred, extra in [
        ("Lightweight-Text", base_pred, {}),
        ("Lightweight+Topic", topic_res.pred, {
            "meta_C": topic_res.c_value, "override_threshold": topic_res.threshold, "override_count": topic_res.override_count,
        }),
        ("Lightweight+Ontology", onto_res.pred, {
            "meta_C": onto_res.c_value, "override_threshold": onto_res.threshold, "override_count": onto_res.override_count,
        }),
    ]:
        m = metric_dict(test["label"], pred)
        row = {"seed": seed, "family": "lightweight", "Model": model_name, **m, **extra}
        rows.append(row)
        cm = confusion_matrix(test["label"], pred, labels=[0, 1, 2])
        pd.DataFrame(cm, index=["gold_neg", "gold_neu", "gold_pos"], columns=["pred_neg", "pred_neu", "pred_pos"]).to_csv(
            seed_dir / f"confusion_{model_name.replace('+','_').replace('-','_')}.csv", encoding="utf-8-sig"
        )
    return rows


def run_phobert_seed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    seed_dir: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    patience: int,
    num_workers: int,
    max_train_samples: Optional[int] = None,
    max_dev_samples: Optional[int] = None,
) -> List[Dict[str, object]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    from sklearn.utils.class_weight import compute_class_weight
    from torch.amp import autocast, GradScaler

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log("[WARNING] PhoBERT is running on CPU. Kaggle/Colab GPU is strongly recommended.")

    idx = np.arange(len(train_df))
    tr_idx, dev_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=train_df["label"]
    )
    tr = train_df.iloc[tr_idx].reset_index(drop=True)
    dev = train_df.iloc[dev_idx].reset_index(drop=True)
    test = test_df.reset_index(drop=True)

    def _stratified_cap(frame: pd.DataFrame, cap: Optional[int], rs: int) -> pd.DataFrame:
        if cap is None or cap <= 0 or len(frame) <= cap:
            return frame.reset_index(drop=True)
        pieces = []
        n_total = len(frame)
        for _, g in frame.groupby("label", sort=False):
            take = max(1, round(cap * len(g) / n_total))
            pieces.append(g.sample(n=min(take, len(g)), random_state=rs))
        out = pd.concat(pieces, axis=0)
        if len(out) > cap:
            out = out.sample(n=cap, random_state=rs)
        elif len(out) < cap:
            remaining = frame.drop(index=out.index, errors="ignore")
            if len(remaining):
                add_n = min(cap - len(out), len(remaining))
                out = pd.concat([out, remaining.sample(n=add_n, random_state=rs)], axis=0)
        return out.sample(frac=1.0, random_state=rs).reset_index(drop=True)

    tr = _stratified_cap(tr, max_train_samples, seed)
    dev = _stratified_cap(dev, max_dev_samples, seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class TextDataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.df = df.reset_index(drop=True)
            enc = tokenizer(
                self.df["sentence"].tolist(),
                padding="max_length", truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = torch.tensor(self.df["label"].to_numpy(), dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {
                "input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels": self.labels[i],
                "index": torch.tensor(i, dtype=torch.long),
            }

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size
            self.dropout = nn.Dropout(0.30)
            self.classifier = nn.Linear(hidden, 3)

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state[:, 0, :]
            return self.classifier(self.dropout(h))

    train_ds, dev_ds, test_ds = TextDataset(tr), TextDataset(dev), TextDataset(test)
    gen = torch.Generator(); gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    dev_loader = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = PhoBERTClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(total_steps * warmup_ratio)), num_training_steps=total_steps
    )
    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=tr["label"].to_numpy())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    def infer(loader):
        model.eval()
        all_logits, all_y, all_idx = [], [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    logits = model(ids, mask)
                all_logits.append(logits.float().cpu().numpy())
                all_y.append(batch["labels"].numpy())
                all_idx.append(batch["index"].numpy())
        return np.vstack(all_logits), np.concatenate(all_y), np.concatenate(all_idx)

    best_state = None
    best_f1 = -1.0
    bad_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(ids, mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item())

        dev_logits, dev_y, _ = infer(dev_loader)
        dev_pred = dev_logits.argmax(axis=1)
        dev_metrics = metric_dict(dev_y, dev_pred)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)),
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        })
        log(f"[seed={seed}] epoch={epoch} loss={history[-1]['train_loss']:.4f} dev_macro_f1={dev_metrics['Macro-F1']:.4f}")

        if dev_metrics["Macro-F1"] > best_f1 + 1e-6:
            best_f1 = dev_metrics["Macro-F1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            log(f"[seed={seed}] early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(seed_dir / "phobert_training_history.csv", index=False, encoding="utf-8-sig")
    if best_state is not None:
        model.load_state_dict(best_state)

    dev_logits, dev_y, _ = infer(dev_loader)
    test_logits, test_y, _ = infer(test_loader)
    base_pred = test_logits.argmax(axis=1)

    # Topic-only and full ontology meta-classifier ablations.
    builder = OntologyLexicalFeatureBuilder().fit(tr)
    dev_topic = builder.topic_metadata(dev)
    test_topic = builder.topic_metadata(test)
    dev_onto = builder.transform(dev)
    test_onto = builder.transform(test)

    topic_res = fit_conservative_meta(dev_logits, dev_topic, dev_y, test_logits, test_topic, seed)
    onto_res = fit_conservative_meta(dev_logits, dev_onto, dev_y, test_logits, test_onto, seed)

    rows = []
    for model_label, pred, extra in [
        ("PhoBERT-Text", base_pred, {}),
        ("PhoBERT+Topic", topic_res.pred, {
            "meta_C": topic_res.c_value, "override_threshold": topic_res.threshold, "override_count": topic_res.override_count,
        }),
        ("PhoBERT+Ontology", onto_res.pred, {
            "meta_C": onto_res.c_value, "override_threshold": onto_res.threshold, "override_count": onto_res.override_count,
        }),
    ]:
        m = metric_dict(test_y, pred)
        rows.append({"seed": seed, "family": "phobert", "Model": model_label, **m, **extra})
        cm = confusion_matrix(test_y, pred, labels=[0, 1, 2])
        pd.DataFrame(cm, index=["gold_neg", "gold_neu", "gold_pos"], columns=["pred_neg", "pred_neu", "pred_pos"]).to_csv(
            seed_dir / f"confusion_{model_label.replace('+','_').replace('-','_')}.csv", encoding="utf-8-sig"
        )

    pred_out = test[["sentence", "sentiment", "topic"]].copy()
    pred_out["gold"] = test_y
    pred_out["PhoBERT_Text_pred"] = base_pred
    pred_out["PhoBERT_Topic_pred"] = topic_res.pred
    pred_out["PhoBERT_Ontology_pred"] = onto_res.pred
    pred_out.to_csv(seed_dir / "phobert_predictions.csv", index=False, encoding="utf-8-sig")

    # Release GPU memory before next seed.
    del model, train_loader, dev_loader, test_loader, train_ds, dev_ds, test_ds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def extract_ontology_stats(input_dir: Path, output_dir: Path) -> Optional[pd.DataFrame]:
    path = input_dir / "Ontology_NTU_FeedbackSystem_V8.owl"
    if not path.exists():
        return None
    from rdflib import Graph, RDF, OWL, Namespace
    from rdflib.namespace import SKOS
    g = Graph(); g.parse(path)
    stats = [
        ("named_classes", len(set(g.subjects(RDF.type, OWL.Class)))),
        ("object_properties", len(set(g.subjects(RDF.type, OWL.ObjectProperty)))),
        ("datatype_properties", len(set(g.subjects(RDF.type, OWL.DatatypeProperty)))),
        ("named_individuals", len(set(g.subjects(RDF.type, OWL.NamedIndividual)))),
        ("rdf_triples", len(g)),
        ("skos_altLabel_assertions", sum(1 for _ in g.triples((None, SKOS.altLabel, None)))),
    ]
    out = pd.DataFrame(stats, columns=["statistic", "value"])
    out["source_file"] = path.name
    out["paper_use_status"] = "CONDITIONAL: reconcile this V8 ontology schema with the CSoNet manuscript schema before replacing the red warning."
    out.to_csv(output_dir / "ontology_kg_statistics.csv", index=False, encoding="utf-8-sig")
    return out


def extract_iaa(input_dir: Path, output_dir: Path) -> Optional[pd.DataFrame]:
    path = input_dir / "IAA_200_Results.xlsx"
    if not path.exists():
        return None
    df = pd.read_excel(path, sheet_name="Kappa_Summary")
    df.to_csv(output_dir / "human_annotation_iaa.csv", index=False, encoding="utf-8-sig")
    return df


def extract_sparql_examples(input_dir: Path, output_dir: Path) -> Optional[pd.DataFrame]:
    path = input_dir / "Ket_Qua_SPARQL.xlsx"
    if not path.exists():
        return None
    rows = []
    xls = pd.ExcelFile(path)
    if "Q7_TyLeSentiment" in xls.sheet_names:
        q7 = pd.read_excel(path, sheet_name="Q7_TyLeSentiment").head(5)
        for _, r in q7.iterrows():
            rows.append({
                "query": "Q7_TyLeSentiment",
                "course_or_entity": str(r.get("hoc_phan", "")),
                "topic": "",
                "count_or_distribution": f"Negative={r.get('Negative','')}; Neutral={r.get('Neutral','')}; Positive={r.get('Positive','')}; Total={r.get('Tong','')}",
                "supporting_comment": "",
            })
    if "Q2b_TieuCucTheoGV_Detail" in xls.sheet_names:
        q2 = pd.read_excel(path, sheet_name="Q2b_TieuCucTheoGV_Detail").head(5)
        for i, (_, r) in enumerate(q2.iterrows(), start=1):
            rows.append({
                "query": "Q2b_TieuCucTheoGV_Detail",
                "course_or_entity": f"Lecturer_{i:02d}",
                "topic": str(r.get("topic", "")),
                "count_or_distribution": str(r.get("emotion", "")),
                "supporting_comment": str(r.get("noiDung", "")),
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "sparql_representative_results.csv", index=False, encoding="utf-8-sig")
    return out


def summarize_metrics(all_metrics: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ["Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"]
    if all_metrics.empty:
        empty = pd.DataFrame(columns=["Model"])
        empty.to_csv(output_dir / "summary_mean_std.csv", index=False)
        empty.to_csv(output_dir / "paper_ready_table.csv", index=False)
        return empty, empty

    grp = all_metrics.groupby(["family", "Model"])[metric_cols]
    mean = grp.mean()
    std = grp.std(ddof=1)
    count = grp.count()
    rows = []
    for idx in mean.index:
        fam, model = idx
        row = {"family": fam, "Model": model, "n_runs": int(count.loc[idx, "Accuracy"])}
        for m in metric_cols:
            row[f"{m}_mean"] = float(mean.loc[idx, m])
            row[f"{m}_std"] = float(std.loc[idx, m]) if not np.isnan(std.loc[idx, m]) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False, encoding="utf-8-sig")

    paper = summary[["family", "Model", "n_runs"]].copy()
    for m in metric_cols:
        paper[m] = summary.apply(
            lambda r: f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}" if pd.notna(r[f"{m}_std"]) else f"{r[f'{m}_mean']:.4f}",
            axis=1,
        )
    # Add delta vs family baseline when available.
    delta = []
    for _, r in summary.iterrows():
        base_name = "PhoBERT-Text" if r["family"] == "phobert" else "Lightweight-Text"
        base = summary[(summary["family"] == r["family"]) & (summary["Model"] == base_name)]
        if len(base):
            delta.append(float(r["Macro-F1_mean"] - base.iloc[0]["Macro-F1_mean"]))
        else:
            delta.append(np.nan)
    paper["Delta_Macro-F1_vs_family_text"] = [f"{d:+.4f}" if pd.notna(d) else "" for d in delta]
    paper.to_csv(output_dir / "paper_ready_table.csv", index=False, encoding="utf-8-sig")
    return summary, paper


def write_camera_ready_notes(
    train: pd.DataFrame,
    val: pd.DataFrame,
    summary: pd.DataFrame,
    ontology_stats: Optional[pd.DataFrame],
    iaa: Optional[pd.DataFrame],
    output_dir: Path,
    cfg: Dict[str, object],
) -> None:
    def scount(df, label): return int((df["sentiment"] == label).sum())
    lines = []
    lines.append("# Camera-ready replacement notes\n")
    lines.append("## 1. DATA CHECK — ready to replace\n")
    lines.append(
        f"The synthetic benchmark contains {len(train)+len(val):,} instances: {len(train):,} in synthetic_train.csv and {len(val):,} in synthetic_val.csv. "
        f"The training split contains {scount(train,'negative'):,} Negative, {scount(train,'neutral'):,} Neutral, and {scount(train,'positive'):,} Positive instances; "
        f"the validation split contains {scount(val,'negative'):,} Negative, {scount(val,'neutral'):,} Neutral, and {scount(val,'positive'):,} Positive instances. "
        "The final validation split is held fixed across runs, while each seed creates a stratified 80/20 model-training/internal-development split from synthetic_train.csv.\n"
    )
    lines.append("**Caution:** The supplied CSV files do not by themselves document the exact derivation procedure from the official UIT-VSFC release. Keep the wording 'UIT-VSFC-style/derived synthetic benchmark' unless that provenance is independently documented.\n")

    lines.append("## 2. STRONG PHOBERT BASELINE — replace after the five-seed run completes\n")
    lines.append(
        f"PhoBERT configuration: model={cfg['model_name']}; tokenizer=AutoTokenizer for the same checkpoint; max sequence length={cfg['max_length']}; "
        f"classification head=dropout(0.30)+linear(768→3); optimizer=AdamW; learning rate={cfg['lr']}; batch size={cfg['batch_size']}; "
        f"maximum epochs={cfg['epochs']}; weight decay={cfg['weight_decay']}; warm-up ratio={cfg['warmup_ratio']}; "
        f"checkpoint selection=best internal-development Macro-F1 with early stopping patience={cfg['patience']}; seeds={cfg['seeds']}.\n"
    )
    if not summary.empty and (summary["family"] == "phobert").any():
        ph = summary[summary["family"] == "phobert"]
        lines.append("Five-seed summary:\n")
        for _, r in ph.iterrows():
            lines.append(
                f"- {r['Model']}: Accuracy {r['Accuracy_mean']:.4f} ± {r['Accuracy_std']:.4f}; "
                f"Macro-Precision {r['Macro-Precision_mean']:.4f} ± {r['Macro-Precision_std']:.4f}; "
                f"Macro-Recall {r['Macro-Recall_mean']:.4f} ± {r['Macro-Recall_std']:.4f}; "
                f"Macro-F1 {r['Macro-F1_mean']:.4f} ± {r['Macro-F1_std']:.4f}.\n"
            )
    else:
        lines.append("[PENDING: run the full PhoBERT five-seed cell; the script will replace this with exact mean ± std values.]\n")

    lines.append("## 3. RESULT TABLE — automatically generated\n")
    lines.append("Use `paper_ready_table.csv` for the numerical table required by the reviewer. `all_seed_metrics.csv` contains every seed-level result and `summary_mean_std.csv` contains numeric means and standard deviations.\n")

    lines.append("## 4. ONTOLOGY/KG STATISTICS — conditional\n")
    if ontology_stats is not None:
        vals = dict(zip(ontology_stats["statistic"], ontology_stats["value"]))
        lines.append(
            f"The supplied Ontology_NTU_FeedbackSystem_V8.owl contains {vals.get('named_classes')} named classes, {vals.get('object_properties')} object properties, "
            f"{vals.get('datatype_properties')} datatype properties, {vals.get('named_individuals')} named individuals, {vals.get('rdf_triples')} RDF triples, "
            f"and {vals.get('skos_altLabel_assertions')} skos:altLabel assertions.\n"
        )
        lines.append("**Do not insert these counts into the paper until the V8 ontology schema is reconciled with the manuscript's StudentFeedbackForm/FeedbackEntry/TeachingCriterion/LearningOutcome schema.**\n")
    else:
        lines.append("[PENDING ontology file.]\n")

    lines.append("## 5. HUMAN-ANNOTATION EVIDENCE — partially ready\n")
    if iaa is not None:
        for _, r in iaa.iterrows():
            lines.append(f"- {r['task']}: n={int(r['n_samples'])}, observed agreement={r['po_observed']}, Cohen's κ={float(r['cohen_kappa']):.4f}.\n")
        lines.append("These data support annotation reliability for Sentiment/Emotion/Topic, but the supplied file does not contain adjudicated gold labels for TeachingCriterion, perceived LearningOutcome, or ImprovementSuggestion. Keep that part of the reviewer warning until those dimensions are annotated and evaluated.\n")

    lines.append("## 6. SPARQL RESULTS — ready as practical evidence after schema consistency check\n")
    lines.append("Representative de-identified rows are exported to `sparql_representative_results.csv`. They can support a compact paper table once the queried KG is confirmed to be the same final KG described in the manuscript.\n")

    (output_dir / "camera_ready_replacement_notes.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="input")
    ap.add_argument("--output_dir", type=str, default="outputs")
    ap.add_argument("--run_phobert", action="store_true", help="Fine-tune PhoBERT for all requested seeds.")
    ap.add_argument("--skip_lightweight", action="store_true", help="Skip the fast lightweight replication.")
    ap.add_argument("--preflight_only", action="store_true", help="Only export dataset/support statistics; do not run models.")
    ap.add_argument("--force", action="store_true", help="Re-run seeds even when a completed seed JSON exists.")
    ap.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--max_train_samples", type=int, default=None, help="Smoke-test only; subsample the internal train partition.")
    ap.add_argument("--max_dev_samples", type=int, default=None, help="Smoke-test only; subsample the internal dev partition.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seeds").mkdir(exist_ok=True)

    cfg = vars(args).copy()
    cfg["input_dir"] = str(input_dir)
    cfg["output_dir"] = str(output_dir)
    (output_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    train, val = load_synthetic(input_dir)
    stats = export_dataset_statistics(train, val, output_dir)
    ontology_stats = extract_ontology_stats(input_dir, output_dir)
    iaa = extract_iaa(input_dir, output_dir)
    extract_sparql_examples(input_dir, output_dir)

    log(f"[DATA] train={len(train)} val={len(val)}")
    log(f"[DATA] train sentiment counts: {train['sentiment'].value_counts().to_dict()}")
    log(f"[DATA] val sentiment counts: {val['sentiment'].value_counts().to_dict()}")

    if args.preflight_only:
        empty = pd.DataFrame()
        summary, _ = summarize_metrics(empty, output_dir)
        write_camera_ready_notes(train, val, summary, ontology_stats, iaa, output_dir, cfg)
        log(f"Preflight outputs saved to {output_dir}")
        return

    rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        seed_dir = output_dir / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        complete_path = seed_dir / "completed_metrics.json"
        cached_rows = []
        if complete_path.exists() and not args.force:
            cached_rows = json.loads(complete_path.read_text(encoding="utf-8"))

        cached_models = {r.get("Model") for r in cached_rows}
        need_lightweight = (not args.skip_lightweight) and not {
            "Lightweight-Text", "Lightweight+Topic", "Lightweight+Ontology"
        }.issubset(cached_models)
        need_phobert = args.run_phobert and not {
            "PhoBERT-Text", "PhoBERT+Topic", "PhoBERT+Ontology"
        }.issubset(cached_models)

        if cached_rows and not need_lightweight and not need_phobert:
            log(f"[seed={seed}] loading cached completed metrics")
            rows.extend(cached_rows)
            continue

        seed_rows = list(cached_rows)
        seed_everything(seed)
        if need_lightweight:
            log(f"[seed={seed}] running lightweight replication")
            seed_rows = [r for r in seed_rows if r.get("family") != "lightweight"]
            seed_rows.extend(run_lightweight_seed(train, val, seed, seed_dir))
        if need_phobert:
            log(f"[seed={seed}] running PhoBERT")
            seed_rows = [r for r in seed_rows if r.get("family") != "phobert"]
            seed_rows.extend(run_phobert_seed(
                train, val, seed, seed_dir,
                model_name=args.model_name,
                max_length=args.max_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                patience=args.patience,
                num_workers=args.num_workers,
                max_train_samples=args.max_train_samples,
                max_dev_samples=args.max_dev_samples,
            ))
        complete_path.write_text(json.dumps(seed_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.extend(seed_rows)

        # Incremental master save so quota/session loss does not erase completed seeds.
        pd.DataFrame(rows).to_csv(output_dir / "all_seed_metrics_partial.csv", index=False, encoding="utf-8-sig")

    all_metrics = pd.DataFrame(rows)
    all_metrics.to_csv(output_dir / "all_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summary, paper = summarize_metrics(all_metrics, output_dir)
    write_camera_ready_notes(train, val, summary, ontology_stats, iaa, output_dir, cfg)

    manifest = {
        "outputs": [
            "all_seed_metrics.csv",
            "summary_mean_std.csv",
            "paper_ready_table.csv",
            "dataset_statistics.csv",
            "ontology_kg_statistics.csv",
            "human_annotation_iaa.csv",
            "sparql_representative_results.csv",
            "camera_ready_replacement_notes.md",
            "run_config.json",
        ],
        "seeds": args.seeds,
        "phobert_ran": bool(args.run_phobert),
        "lightweight_ran": not args.skip_lightweight,
    }
    (output_dir / "results_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nDONE. Outputs saved to: {output_dir}")
    if not args.run_phobert:
        log("NOTE: PhoBERT was not run. Re-run with --run_phobert on a GPU runtime for the camera-ready baseline.")


if __name__ == "__main__":
    main()
