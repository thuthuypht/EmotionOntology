import argparse
import os
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from rdflib import Graph, RDF, RDFS, OWL, URIRef, Namespace

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel


# --------------------------
# Constants
# --------------------------
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
TOPIC_MAIN = ["Facilities", "TeachingMethod", "CourseContent", "Material", "LecturerAttitude"]
TOPIC_ALL = TOPIC_MAIN + ["OtherTopic"]          # 6 topics
SENT_LIST = ["Negative", "Neutral", "Positive"]  # 3 sentiments


# --------------------------
# Utils
# --------------------------
def clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
    }


def tokenize_basic(text: str) -> List[str]:
    # VN + EN tokenization for ontology matching
    return re.findall(r"[a-zA-ZÀ-ỹđĐ]+", text.lower())


def ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def label_ngram_set(labels: List[str], max_n: int = 3) -> Dict[int, set]:
    sets = {1: set(), 2: set(), 3: set()}
    for lab in labels:
        toks = tokenize_basic(lab)
        if not toks:
            continue
        for n in range(1, min(max_n, len(toks)) + 1):
            for ng in ngrams(toks, n):
                if len(ng) >= 3:
                    sets[n].add(ng)
    return sets


def ngram_counts_for_text(text: str, ng_sets: Dict[int, set]) -> float:
    toks = tokenize_basic(text)
    if not toks:
        return 0.0
    feats1 = set(ngrams(toks, 1))
    feats2 = set(ngrams(toks, 2)) if len(toks) >= 2 else set()
    feats3 = set(ngrams(toks, 3)) if len(toks) >= 3 else set()
    return float(len(feats1 & ng_sets[1]) + len(feats2 & ng_sets[2]) + len(feats3 & ng_sets[3]))


def ensure_exists(p: str) -> None:
    if not Path(p).exists():
        here = Path.cwd()
        print("\n[ERROR] File not found:", p)
        print("Current working directory:", str(here))
        print("Files in this folder:")
        for f in sorted(here.glob("*")):
            print("  -", f.name)
        raise FileNotFoundError(p)


# --------------------------
# Ontology matcher -> o(x) in R^24
# o = topic_counts(6) + sentiment_counts(3) + cross(main_topics x sentiment)(15) = 24
# --------------------------
class OntologyMatcher:
    def __init__(self, owl_path: str):
        ensure_exists(owl_path)
        self.g = Graph()
        self.g.parse(owl_path)

        self.ont_iri = None
        for s in self.g.subjects(RDF.type, OWL.Ontology):
            self.ont_iri = str(s)
            break
        if self.ont_iri is None:
            raise ValueError("Cannot find owl:Ontology IRI in the OWL file.")

        self.base = self.ont_iri + "#"
        self.SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

        # Prebuild ngram sets for each concept
        self.topic_ng = {tp: label_ngram_set(self._get_labels(tp), max_n=3) for tp in TOPIC_ALL}
        self.sent_ng = {s: label_ngram_set(self._get_labels(s), max_n=3) for s in SENT_LIST}

    def _get_labels(self, local: str) -> List[str]:
        ind = URIRef(self.base + local)
        labels = set()
        for lit in self.g.objects(ind, self.SKOS.altLabel):
            labels.add(str(lit).lower())
        for lit in self.g.objects(ind, RDFS.label):
            labels.add(str(lit).lower())
        return sorted(labels, key=lambda x: (-len(x), x))

    def build_o(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        topic_counts = np.zeros((n, len(TOPIC_ALL)), dtype=np.float32)
        sent_counts = np.zeros((n, 3), dtype=np.float32)  # [neg, neu, pos]

        for i, t in enumerate(texts):
            for j, tp in enumerate(TOPIC_ALL):
                topic_counts[i, j] = ngram_counts_for_text(t, self.topic_ng[tp])
            sent_counts[i, 0] = ngram_counts_for_text(t, self.sent_ng["Negative"])
            sent_counts[i, 1] = ngram_counts_for_text(t, self.sent_ng["Neutral"])
            sent_counts[i, 2] = ngram_counts_for_text(t, self.sent_ng["Positive"])

        cross = np.einsum("ij,ik->ijk", topic_counts[:, :len(TOPIC_MAIN)], sent_counts).reshape(n, -1).astype(np.float32)
        o = np.concatenate([topic_counts, sent_counts, cross], axis=1)
        assert o.shape[1] == 24
        return o


# --------------------------
# DataLoader
# --------------------------
class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


# --------------------------
# PhoBERT encoder (CLS pooling)
# --------------------------
class PhoBERTEncoder(nn.Module):
    def __init__(self, model_name: str = "vinai/phobert-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)

    @torch.no_grad()
    def encode(self, texts: List[str], device: torch.device, max_length: int = 128) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        out = self.model(**enc)
        h = out.last_hidden_state[:, 0, :]  # CLS: [B, 768]
        return h


# --------------------------
# Models: baseline and gated fusion
# --------------------------
class BaselineClassifier(nn.Module):
    def __init__(self, h_dim: int = 768, n_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(h_dim, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class GatedFusionClassifier(nn.Module):
    def __init__(self, h_dim: int = 768, o_dim: int = 24, n_classes: int = 3):
        super().__init__()
        self.gate = nn.Linear(h_dim, o_dim)
        self.fc = nn.Linear(h_dim + o_dim, n_classes)

    def forward(self, h: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(h))     # [B, 24]
        z = torch.cat([h, o * g], dim=1)    # [B, 768+24]
        return self.fc(z)


def train_eval_one_seed(
    seed: int,
    train_texts: List[str],
    train_y: np.ndarray,
    test_texts: List[str],
    test_y: np.ndarray,
    matcher: OntologyMatcher,
    use_gated: bool,
    epochs: int,
    lr: float,
    batch_size: int,
    max_len: int,
    device: torch.device
) -> Dict[str, float]:
    set_seed(seed)

    encoder = PhoBERTEncoder("vinai/phobert-base").to(device)
    encoder.eval()  # frozen encoder (fast & stable)

    clf = (GatedFusionClassifier().to(device) if use_gated else BaselineClassifier().to(device))

    # class weights (helps imbalance)
    counts = np.bincount(train_y, minlength=3).astype(np.float32)
    counts[counts == 0] = 1.0
    w = counts.sum() / counts
    w = (w / w.sum()) * 3.0
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    opt = torch.optim.Adam(clf.parameters(), lr=lr)

    ds = SimpleTextDataset(train_texts, train_y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Precompute ontology features for test (train computed per-batch)
    o_test = torch.tensor(matcher.build_o(test_texts), dtype=torch.float32, device=device)

    clf.train()
    for _ in range(epochs):
        for texts_b, y_b in dl:
            y_b = y_b.to(device)
            opt.zero_grad()

            h_b = encoder.encode(texts_b, device=device, max_length=max_len)

            if use_gated:
                o_b = torch.tensor(matcher.build_o(texts_b), dtype=torch.float32, device=device)
                logits = clf(h_b, o_b)
            else:
                logits = clf(h_b)

            loss = loss_fn(logits, y_b)
            loss.backward()
            opt.step()

    clf.eval()
    with torch.no_grad():
        h_te = encoder.encode(test_texts, device=device, max_length=max_len)
        if use_gated:
            logits = clf(h_te, o_test)
        else:
            logits = clf(h_te)
        pred = logits.argmax(dim=1).cpu().numpy()

    return macro_metrics(test_y, pred)


# --------------------------
# Tables + Figures
# --------------------------
def table5b_mean_std(df5a: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"]
    grp = df5a.groupby(["Dataset", "Model"])[metrics]
    mean_df = grp.mean().reset_index()
    std_df = grp.std(ddof=1).reset_index()

    out_rows = []
    for _, r in mean_df.iterrows():
        ds = r["Dataset"]; md = r["Model"]
        rstd = std_df[(std_df["Dataset"] == ds) & (std_df["Model"] == md)].iloc[0]
        row = {"Dataset": ds, "Model": md}
        for m in metrics:
            row[f"{m} (mean ± std)"] = f"{r[m]:.4f} ± {rstd[m]:.4f}"
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def fig5_best_seed(df5a: pd.DataFrame, out_png: str) -> int:
    metrics = ["Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"]
    ours = df5a[df5a["Model"] == "PhoBERT + gated ontology (ours)"].copy()
    score = ours.groupby("Seed")[["Accuracy", "Macro-F1"]].mean().mean(axis=1)
    best_seed = int(score.idxmax())

    sub = df5a[df5a["Seed"] == best_seed].copy()
    dataset_order = ["Benchmark split (UIT-VSFC-style)", "NTU survey (silver labels)"]
    model_order = ["PhoBERT-text baseline", "PhoBERT + gated ontology (ours)"]

    sub["Dataset"] = pd.Categorical(sub["Dataset"], categories=dataset_order, ordered=True)
    sub["Model"] = pd.Categorical(sub["Model"], categories=model_order, ordered=True)
    sub = sub.sort_values(["Dataset", "Model"])

    groups = [(ds, md) for ds in dataset_order for md in model_order]
    x = np.arange(len(groups))
    bar_w = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w

    def get_val(ds, md, metric):
        r = sub[(sub["Dataset"] == ds) & (sub["Model"] == md)]
        return float(r.iloc[0][metric])

    fig, ax = plt.subplots(figsize=(14, 6.2))
    for i, m in enumerate(metrics):
        vals = np.array([get_val(ds, md, m) for ds, md in groups])
        bars = ax.bar(x + offsets[i], vals, width=bar_w, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds}\n{md}" for ds, md in groups], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Fig. 5. Best-seed results (Seed {best_seed}) for PhoBERT + Gated Ontology Fusion")
    ax.legend(ncol=4, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    plt.savefig(out_png, dpi=420, bbox_inches="tight")
    plt.close(fig)

    return best_seed


def fig6_mean_std(df5b: pd.DataFrame, out_png: str) -> None:
    metrics = ["Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"]

    # parse "mean ± std"
    def parse_ms(s: str) -> Tuple[float, float]:
        parts = re.split(r"\s*±\s*", str(s))
        return float(parts[0]), float(parts[1])

    rows = []
    for _, r in df5b.iterrows():
        out = {"Dataset": r["Dataset"], "Model": r["Model"]}
        for m in metrics:
            mu, sd = parse_ms(r[f"{m} (mean ± std)"])
            out[f"{m}_mean"] = mu
            out[f"{m}_std"] = sd
        rows.append(out)
    d = pd.DataFrame(rows)

    dataset_order = ["Benchmark split (UIT-VSFC-style)", "NTU survey (silver labels)"]
    model_order = ["PhoBERT-text baseline", "PhoBERT + gated ontology (ours)"]

    d["Dataset"] = pd.Categorical(d["Dataset"], categories=dataset_order, ordered=True)
    d["Model"] = pd.Categorical(d["Model"], categories=model_order, ordered=True)
    d = d.sort_values(["Dataset", "Model"]).reset_index(drop=True)

    groups = [(ds, md) for ds in dataset_order for md in model_order]
    x = np.arange(len(groups))
    bar_w = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w

    def get_row(ds, md):
        return d[(d["Dataset"] == ds) & (d["Model"] == md)].iloc[0]

    fig, ax = plt.subplots(figsize=(18, 6.5))
    for i, m in enumerate(metrics):
        means = np.array([get_row(ds, md)[f"{m}_mean"] for ds, md in groups])
        stds = np.array([get_row(ds, md)[f"{m}_std"] for ds, md in groups])

        bars = ax.bar(x + offsets[i], means, yerr=stds, capsize=4, width=bar_w, label=m)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds}\n{md}" for ds, md in groups], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Fig. 6. Mean ± std over 5 seeds for PhoBERT + Gated Ontology Fusion (combined)")
    ax.legend(ncol=4, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    plt.savefig(out_png, dpi=420, bbox_inches="tight")
    plt.close(fig)


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", required=True, help="OWL file (e.g., EONLP_V10_skos_enriched_final.owl)")
    ap.add_argument("--bench_train", required=True, help="Benchmark train CSV")
    ap.add_argument("--bench_val", required=True, help="Benchmark val CSV")
    ap.add_argument("--ntu", required=True, help="NTU dataset CSV (with y or silver_sentiment)")
    ap.add_argument("--out_dir", default="outputs_phobert_gated", help="Output folder")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5")
    args = ap.parse_args()

    ensure_exists(args.owl)
    ensure_exists(args.bench_train)
    ensure_exists(args.bench_val)
    ensure_exists(args.ntu)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CWD:", str(Path.cwd()))

    # Load benchmark
    bench_tr = pd.read_csv(args.bench_train)
    bench_va = pd.read_csv(args.bench_val)

    text_col = "sentence" if "sentence" in bench_tr.columns else ("text" if "text" in bench_tr.columns else bench_tr.columns[0])
    bench_tr["text"] = bench_tr[text_col].map(clean_text)
    bench_va["text"] = bench_va[text_col].map(clean_text)

    bench_tr["y"] = bench_tr["sentiment"].astype(str).str.lower().map(LABEL_MAP).to_numpy()
    bench_va["y"] = bench_va["sentiment"].astype(str).str.lower().map(LABEL_MAP).to_numpy()

    if "topic_owl" not in bench_tr.columns:
        # fallback mapping
        topic_map = {"facility": "Facilities", "curriculum": "CourseContent", "lecturer": "TeachingMethod", "others": "OtherTopic"}
        bench_tr["topic_owl"] = bench_tr["topic"].astype(str).str.lower().map(topic_map).fillna("OtherTopic")
        bench_va["topic_owl"] = bench_va["topic"].astype(str).str.lower().map(topic_map).fillna("OtherTopic")

    # Load NTU
    ntu = pd.read_csv(args.ntu)
    ntu_text_col = "text" if "text" in ntu.columns else ("text_clean" if "text_clean" in ntu.columns else ntu.columns[0])
    ntu["text"] = ntu[ntu_text_col].map(clean_text)

    if "y" in ntu.columns:
        ntu["y"] = ntu["y"].astype(int)
    elif "silver_sentiment" in ntu.columns:
        ntu["y"] = ntu["silver_sentiment"].astype(str).str.lower().map(LABEL_MAP).fillna(1).astype(int)
    else:
        print("[WARN] NTU file has no 'y' or 'silver_sentiment'. Using all-neutral labels (not recommended).")
        ntu["y"] = 1

    matcher = OntologyMatcher(args.owl)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for seed in seeds:
        # Benchmark: fixed train/val
        m_base = train_eval_one_seed(
            seed, bench_tr["text"].tolist(), bench_tr["y"].to_numpy(),
            bench_va["text"].tolist(), bench_va["y"].to_numpy(),
            matcher, use_gated=False,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, max_len=args.max_len, device=device
        )
        m_ours = train_eval_one_seed(
            seed, bench_tr["text"].tolist(), bench_tr["y"].to_numpy(),
            bench_va["text"].tolist(), bench_va["y"].to_numpy(),
            matcher, use_gated=True,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, max_len=args.max_len, device=device
        )
        rows.append([seed, "Benchmark split (UIT-VSFC-style)", "PhoBERT-text baseline", m_base["acc"], m_base["prec"], m_base["rec"], m_base["f1"]])
        rows.append([seed, "Benchmark split (UIT-VSFC-style)", "PhoBERT + gated ontology (ours)", m_ours["acc"], m_ours["prec"], m_ours["rec"], m_ours["f1"]])

        # NTU: per-seed 80/20 split
        idx = np.arange(len(ntu))
        tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=ntu["y"])
        nt_tr_text = ntu.loc[tr_idx, "text"].tolist()
        nt_tr_y = ntu.loc[tr_idx, "y"].to_numpy()
        nt_te_text = ntu.loc[te_idx, "text"].tolist()
        nt_te_y = ntu.loc[te_idx, "y"].to_numpy()

        m_base_n = train_eval_one_seed(
            seed, nt_tr_text, nt_tr_y, nt_te_text, nt_te_y,
            matcher, use_gated=False,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, max_len=args.max_len, device=device
        )
        m_ours_n = train_eval_one_seed(
            seed, nt_tr_text, nt_tr_y, nt_te_text, nt_te_y,
            matcher, use_gated=True,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, max_len=args.max_len, device=device
        )
        rows.append([seed, "NTU survey (silver labels)", "PhoBERT-text baseline", m_base_n["acc"], m_base_n["prec"], m_base_n["rec"], m_base_n["f1"]])
        rows.append([seed, "NTU survey (silver labels)", "PhoBERT + gated ontology (ours)", m_ours_n["acc"], m_ours_n["prec"], m_ours_n["rec"], m_ours_n["f1"]])

    df5a = pd.DataFrame(rows, columns=["Seed", "Dataset", "Model", "Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"])
    table5a_path = out_dir / "Table5A_PhoBERT_GatedFusion.csv"
    df5a.to_csv(table5a_path, index=False)
    print("Saved:", table5a_path)

    df5b = table5b_mean_std(df5a)
    table5b_path = out_dir / "Table5B_PhoBERT_GatedFusion_MeanStd.csv"
    df5b.to_csv(table5b_path, index=False, encoding="utf-8-sig")
    print("Saved:", table5b_path)

    fig5_path = out_dir / "Fig5_PhoBERT_GatedFusion_BestSeed.png"
    best_seed = fig5_best_seed(df5a, str(fig5_path))
    print("Saved:", fig5_path, "(best seed:", best_seed, ")")

    fig6_path = out_dir / "Fig6_PhoBERT_GatedFusion_MeanStd_Combined.png"
    fig6_mean_std(df5b, str(fig6_path))
    print("Saved:", fig6_path)


if __name__ == "__main__":
    main()