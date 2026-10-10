# run_experiments_table5A_5B.py
# Reproduce OWL enrichment + Table 5A (per-seed) + Table 5B (mean±std)
#
# Inputs expected in the same folder (or provide full paths):
#   - EONLP_V5_with_vi_labels.owl
#   - synthetic_train.csv
#   - synthetic_val.csv
#   - Khoa CNTT Gop y.xlsx
#
# Outputs:
#   - EONLP_V10_skos_enriched_final.owl
#   - NTU_feedback_updated_v2.csv
#   - Table5A_PerSeed_Final.csv
#   - Table5B_MeanStd_Final.csv
#
# Requirements:
#   pip install pandas numpy rdflib scikit-learn scipy openpyxl matplotlib

import os, re, math
from collections import Counter
import numpy as np
import pandas as pd

from rdflib import Graph, RDF, RDFS, OWL, URIRef, Literal, Namespace

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from scipy.sparse import hstack, csr_matrix


# ----------------------------
# Paths (edit if needed)
# ----------------------------
OWL_IN = "EONLP_V5_with_vi_labels.owl"
BENCH_TRAIN = "synthetic_train.csv"
BENCH_VAL = "synthetic_val.csv"
NTU_EXCEL = "Khoa CNTT Gop y.xlsx"

OWL_OUT = "EONLP_V10_skos_enriched_final.owl"
NTU_OUT_CSV = "NTU_feedback_updated_v2.csv"
TABLE5A_OUT = "Table5A_PerSeed_Final.csv"
TABLE5B_OUT = "Table5B_MeanStd_Final.csv"

SEEDS = [1, 2, 3, 4, 5]


# ----------------------------
# Helpers
# ----------------------------
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

TOPIC_LIST = [
    "Facilities", "TeachingMethod", "CourseContent",
    "Material", "LecturerAttitude", "OtherTopic"
]
TOPIC_MAP_BENCH = {
    "facility": "Facilities",
    "curriculum": "CourseContent",
    "lecturer": "TeachingMethod",
    "others": "OtherTopic"
}


def clean_text(s):
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def tokenize_basic(text: str):
    return re.findall(r"[a-zA-ZÀ-ỹđĐ]+", text.lower())


def ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def has_vi_diacritics(s: str) -> bool:
    return bool(re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", s))


def macro_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1


def fmt4(x):
    return float(f"{x:.4f}")


# ----------------------------
# 1) Load benchmark data
# ----------------------------
bench_tr = pd.read_csv(BENCH_TRAIN)
bench_va = pd.read_csv(BENCH_VAL)

bench_tr["text"] = bench_tr["sentence"].map(clean_text)
bench_va["text"] = bench_va["sentence"].map(clean_text)

bench_tr["y"] = bench_tr["sentiment"].astype(str).str.lower().map(LABEL_MAP).to_numpy()
bench_va["y"] = bench_va["sentiment"].astype(str).str.lower().map(LABEL_MAP).to_numpy()

bench_tr["topic_owl"] = bench_tr["topic"].astype(str).str.lower().map(TOPIC_MAP_BENCH).fillna("OtherTopic")
bench_va["topic_owl"] = bench_va["topic"].astype(str).str.lower().map(TOPIC_MAP_BENCH).fillna("OtherTopic")


# ----------------------------
# 2) Enrich OWL with skos:altLabel (curated + data-driven)
# ----------------------------
g = Graph()
g.parse(OWL_IN)

ont_iri = None
for s in g.subjects(RDF.type, OWL.Ontology):
    ont_iri = str(s)
    break
base = ont_iri + "#"

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
g.bind("skos", SKOS)


def add_altlabel(local, text, lang=None):
    ind = URIRef(base + local)
    lit = Literal(text, lang=lang) if lang else Literal(text)
    if (ind, SKOS.altLabel, lit) not in g:
        g.add((ind, SKOS.altLabel, lit))


# curated topic altLabels
cur_topic = {
    "Facilities": {"vi": ["cơ sở vật chất", "phòng học", "phòng máy", "máy chiếu", "wifi", "mạng", "thiết bị"],
                   "en": ["facility", "facilities", "classroom", "lab", "equipment", "projector", "wifi"]},
    "TeachingMethod": {"vi": ["phương pháp giảng dạy", "cách dạy", "tốc độ giảng", "ví dụ", "minh họa", "thực hành", "tương tác"],
                       "en": ["teaching method", "lecture pace", "examples", "practice", "interaction"]},
    "CourseContent": {"vi": ["nội dung môn học", "chương trình", "đề cương", "kiến thức", "chủ đề"],
                      "en": ["curriculum", "course content", "syllabus", "topics"]},
    "Material": {"vi": ["tài liệu", "giáo trình", "slide", "bài giảng", "nguồn tham khảo"],
                 "en": ["materials", "textbook", "slides", "lecture notes", "references"]},
    "LecturerAttitude": {"vi": ["thái độ giảng viên", "nhiệt tình", "hỗ trợ", "thân thiện", "giải đáp"],
                         "en": ["lecturer attitude", "supportive", "helpful", "friendly", "responsive"]},
    "OtherTopic": {"vi": ["khác", "không có ý kiến"],
                   "en": ["other", "no comment"]},
}

# curated sentiment altLabels
cur_sent = {
    "Positive": {"vi": ["tốt", "tuyệt vời", "hữu ích", "rõ ràng", "dễ hiểu", "hiệu quả", "hài lòng"],
                 "en": ["good", "great", "excellent", "helpful", "clear", "effective", "satisfied"]},
    "Negative": {"vi": ["khó hiểu", "kém", "tệ", "mơ hồ", "không rõ", "không hài lòng", "quá nhanh", "quá chậm", "thiếu"],
                 "en": ["bad", "poor", "confusing", "unclear", "not satisfied", "too fast", "too slow", "lack"]},
    "Neutral": {"vi": ["bình thường", "tạm", "khá", "không ý kiến", "không có ý kiến"],
                "en": ["neutral", "average", "okay", "no opinion"]},
}

for k, langs in cur_topic.items():
    for v in langs.get("vi", []): add_altlabel(k, v, "vi")
    for e in langs.get("en", []): add_altlabel(k, e, "en")

for k, langs in cur_sent.items():
    for v in langs.get("vi", []): add_altlabel(k, v, "vi")
    for e in langs.get("en", []): add_altlabel(k, e, "en")

# data-driven discriminative cues per sentiment class
class_counts = {0: Counter(), 1: Counter(), 2: Counter()}
total_counts = Counter()

for text, y in zip(bench_tr["text"], bench_tr["y"]):
    toks = tokenize_basic(text)
    feats = toks + ngrams(toks, 2)
    for f in feats:
        if len(f) < 3:
            continue
        class_counts[int(y)][f] += 1
        total_counts[f] += 1


def top_log_odds(target_cls, K=160, min_count=10):
    scores = []
    sum_t = sum(class_counts[target_cls].values()) + 1.0
    sum_all = sum(total_counts.values()) + 1.0
    sum_rest = (sum_all - sum_t) + 1.0
    for term, c_t in class_counts[target_cls].items():
        if total_counts[term] < min_count:
            continue
        c_r = total_counts[term] - c_t
        p_t = (c_t + 1.0) / sum_t
        p_r = (c_r + 1.0) / sum_rest
        score = math.log(p_t / p_r)
        scores.append((score, term))
    scores.sort(reverse=True)
    return [t for _, t in scores[:K]]


terms_by_sent = {
    "Negative": top_log_odds(0),
    "Neutral": top_log_odds(1),
    "Positive": top_log_odds(2),
}
for slevel, terms in terms_by_sent.items():
    for term in terms:
        if len(term) > 40:
            continue
        lang = "vi" if (has_vi_diacritics(term) or "đ" in term) else "en"
        add_altlabel(slevel, term, lang)

# data-driven topic phrases
topic_counts = {t: Counter() for t in TOPIC_LIST}
for text, tp in zip(bench_tr["text"], bench_tr["topic_owl"]):
    toks = tokenize_basic(text)
    feats = toks + ngrams(toks, 2)
    for f in feats:
        if len(f) < 3:
            continue
        topic_counts[tp][f] += 1


def top_tf(counter: Counter, K=80, min_count=12):
    items = [(c, t) for t, c in counter.items() if c >= min_count]
    items.sort(reverse=True)
    return [t for _, t in items[:K]]


for tp in ["Facilities", "TeachingMethod", "CourseContent", "OtherTopic"]:
    for term in top_tf(topic_counts[tp]):
        if len(term) > 40:
            continue
        lang = "vi" if (has_vi_diacritics(term) or "đ" in term) else "en"
        add_altlabel(tp, term, lang)

# write OWL
g.serialize(destination=OWL_OUT, format="xml")


# ----------------------------
# 3) Build concept-match feature sets from OWL labels
# ----------------------------
def get_labels(local):
    ind = URIRef(base + local)
    labels = set()
    for lit in g.objects(ind, SKOS.altLabel):
        labels.add(str(lit).lower())
    for lit in g.objects(ind, RDFS.label):
        labels.add(str(lit).lower())
    return labels


topic_labels = {tp: get_labels(tp) for tp in TOPIC_LIST}
sent_labels = {s: get_labels(s) for s in ["Positive", "Negative", "Neutral"]}


def label_ngram_set(labels, max_n=3):
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


topic_ng = {tp: label_ngram_set(topic_labels[tp], max_n=3) for tp in TOPIC_LIST}
sent_ng = {s: label_ngram_set(sent_labels[s], max_n=3) for s in ["Positive", "Negative", "Neutral"]}


def ngram_counts_for_text(text, ng_sets):
    toks = tokenize_basic(text)
    feats = {
        1: set(ngrams(toks, 1)),
        2: set(ngrams(toks, 2)),
        3: set(ngrams(toks, 3)),
    }
    return float(
        len(feats[1].intersection(ng_sets[1])) +
        len(feats[2].intersection(ng_sets[2])) +
        len(feats[3].intersection(ng_sets[3]))
    )


def build_ontology_features(texts, topics_owl=None):
    n = len(texts)
    topic_mc = np.zeros((n, len(TOPIC_LIST)), dtype=np.float32)
    sent_mc = np.zeros((n, 3), dtype=np.float32)  # [neg,neu,pos]

    for i, t in enumerate(texts):
        for j, tp in enumerate(TOPIC_LIST):
            topic_mc[i, j] = ngram_counts_for_text(t, topic_ng[tp])
        neg = ngram_counts_for_text(t, sent_ng["Negative"])
        neu = ngram_counts_for_text(t, sent_ng["Neutral"])
        pos = ngram_counts_for_text(t, sent_ng["Positive"])
        sent_mc[i, :] = [neg, neu, pos]

    # metadata one-hot for benchmark; inferred for NTU
    if topics_owl is not None:
        tp_to_idx = {t: i for i, t in enumerate(TOPIC_LIST)}
        idxs = np.array([tp_to_idx.get(tp, tp_to_idx["OtherTopic"]) for tp in topics_owl], dtype=np.int32)
    else:
        idxs = topic_mc.argmax(axis=1).astype(np.int32)

    rows = np.arange(n)
    data = np.ones(n, dtype=np.float32)
    meta_oh = csr_matrix((data, (rows, idxs)), shape=(n, len(TOPIC_LIST)))
    sent_csr = csr_matrix(sent_mc)
    cross = np.einsum("ij,ik->ijk", meta_oh.toarray(), sent_csr.toarray()).reshape(n, -1).astype(np.float32)

    o = hstack([csr_matrix(topic_mc), sent_csr, meta_oh, csr_matrix(cross)], format="csr")
    return o


# ----------------------------
# 4) Text feature model
# ----------------------------
hv_word = HashingVectorizer(
    analyzer="word", ngram_range=(1, 2),
    n_features=2**16, alternate_sign=False,
    norm=None, lowercase=True
)
hv_char = HashingVectorizer(
    analyzer="char", ngram_range=(3, 5),
    n_features=2**16, alternate_sign=False,
    norm=None, lowercase=True
)


def build_text_X(train_texts, test_texts):
    Xtr = hstack([hv_word.transform(train_texts), hv_char.transform(train_texts)], format="csr")
    Xte = hstack([hv_word.transform(test_texts), hv_char.transform(test_texts)], format="csr")
    tfidf = TfidfTransformer()
    Xtr = tfidf.fit_transform(Xtr)
    Xte = tfidf.transform(Xte)
    return Xtr, Xte


def train_text_clf(Xtr, y, seed):
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=1500,
        tol=1e-3,
        shuffle=True,
        random_state=seed,
        class_weight="balanced",
        n_jobs=1
    )
    clf.fit(Xtr, y)
    return clf


# safe override: only accept meta change when meta confidence >= thr
def meta_override_predict(meta, logits, onto_feats, baseline_pred, thr):
    proba = meta.predict_proba(np.hstack([logits, onto_feats]))
    meta_pred = proba.argmax(axis=1)
    meta_conf = proba.max(axis=1)
    out = baseline_pred.copy()
    mask = (meta_conf >= thr) & (meta_pred != baseline_pred)
    out[mask] = meta_pred[mask]
    return out


def tune_meta(logits_dev, onto_dev, y_dev, seed):
    # split dev into dev_train/dev_val
    idx = np.arange(len(y_dev))
    dtr, dva = train_test_split(idx, test_size=0.4, random_state=seed, stratify=y_dev)

    Xtr = np.hstack([logits_dev[dtr], onto_dev[dtr]])
    ytr = y_dev[dtr]
    Xva = np.hstack([logits_dev[dva], onto_dev[dva]])
    yva = y_dev[dva]

    base_pred_va = logits_dev[dva].argmax(axis=1)
    acc_b, _, _, f1_b = macro_metrics(yva, base_pred_va)

    best = None
    best_score = -1e9
    for C in [0.02, 0.05, 0.1, 0.2, 0.5]:
        meta = LogisticRegression(max_iter=500, multi_class="multinomial", solver="lbfgs", C=C)
        meta.fit(Xtr, ytr)
        for thr in [0.55, 0.6, 0.65, 0.7, 0.75]:
            pred = meta_override_predict(meta, logits_dev[dva], onto_dev[dva], base_pred_va, thr)
            acc, _, _, f1 = macro_metrics(yva, pred)
            if acc > acc_b and f1 > f1_b:
                score = (acc - acc_b) + (f1 - f1_b)
                if score > best_score:
                    best_score = score
                    best = (C, thr)

    if best is None:
        best = (0.1, 0.8)  # conservative fallback
    return best


# ----------------------------
# 5) Load NTU and build silver labels (for evaluation only)
# ----------------------------
ntu = pd.read_excel(NTU_EXCEL, header=7)
ntu["text"] = ntu["Nội dung góp ý"].map(clean_text)
ntu["question"] = ntu["Tên câu hỏi"].map(clean_text)


def silver_label_ntU(question, text):
    if "ưu điểm" in question or "điểm mạnh" in question:
        return 2  # positive
    neg = ngram_counts_for_text(text, sent_ng["Negative"])
    neu = ngram_counts_for_text(text, sent_ng["Neutral"])
    pos = ngram_counts_for_text(text, sent_ng["Positive"])
    arr = np.array([neg, neu, pos], dtype=np.float32)
    if arr.max() > 0:
        return int(arr.argmax())
    return 1  # neutral


ntu["y"] = [silver_label_ntU(q, t) for q, t in zip(ntu["question"], ntu["text"])]
ntu.to_csv(NTU_OUT_CSV, index=False, encoding="utf-8-sig")

ntu_texts = ntu["text"].tolist()
ntu_y = ntu["y"].to_numpy()


# ----------------------------
# 6) Run experiments (Table 5A)
# ----------------------------
rows = []

for seed in SEEDS:
    # ----- Benchmark: train/dev split from benchmark train -----
    idx = np.arange(len(bench_tr))
    tr_idx, dev_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=bench_tr["y"])

    tr_text = bench_tr.loc[tr_idx, "text"].tolist()
    tr_y = bench_tr.loc[tr_idx, "y"].to_numpy()
    tr_topic = bench_tr.loc[tr_idx, "topic_owl"].tolist()

    dev_text = bench_tr.loc[dev_idx, "text"].tolist()
    dev_y = bench_tr.loc[dev_idx, "y"].to_numpy()
    dev_topic = bench_tr.loc[dev_idx, "topic_owl"].tolist()

    val_text = bench_va["text"].tolist()
    val_y = bench_va["y"].to_numpy()
    val_topic = bench_va["topic_owl"].tolist()

    # baseline text-only on benchmark
    Xtr, Xval = build_text_X(tr_text, val_text)
    text_clf = train_text_clf(Xtr, tr_y, seed)
    pred_base = text_clf.predict(Xval)
    mb = macro_metrics(val_y, pred_base)
    rows.append([seed, "Benchmark split (synthetic)", "Text-only baseline", *map(fmt4, mb)])

    # meta tuning on dev
    Xtr_dev, Xdev = build_text_X(tr_text, dev_text)
    text_clf_dev = train_text_clf(Xtr_dev, tr_y, seed)
    logits_dev = text_clf_dev.decision_function(Xdev)
    onto_dev = build_ontology_features(dev_text, dev_topic).toarray()

    C_best, thr_best = tune_meta(logits_dev, onto_dev, dev_y, seed)

    # fit meta on full dev
    meta = LogisticRegression(max_iter=500, multi_class="multinomial", solver="lbfgs", C=C_best)
    meta.fit(np.hstack([logits_dev, onto_dev]), dev_y)

    # safe override prediction on val
    logits_val = text_clf.decision_function(Xval)
    onto_val = build_ontology_features(val_text, val_topic).toarray()
    pred_onto = meta_override_predict(meta, logits_val, onto_val, pred_base, thr_best)

    mo = macro_metrics(val_y, pred_onto)
    rows.append([seed, "Benchmark split (synthetic)", "Ontology-assisted (ours)", *map(fmt4, mo)])

    # ----- NTU: train/test split + inner dev for meta -----
    idxn = np.arange(len(ntu_texts))
    trn_idx, ten_idx = train_test_split(idxn, test_size=0.2, random_state=seed, stratify=ntu_y)

    trn_text = [ntu_texts[i] for i in trn_idx]
    trn_y = ntu_y[trn_idx]
    ten_text = [ntu_texts[i] for i in ten_idx]
    ten_y = ntu_y[ten_idx]

    # baseline on NTU
    Xn_tr, Xn_te = build_text_X(trn_text, ten_text)
    text_n = train_text_clf(Xn_tr, trn_y, seed)
    pred_n_base = text_n.predict(Xn_te)
    mnb = macro_metrics(ten_y, pred_n_base)
    rows.append([seed, "NTU survey (silver labels)", "Text-only baseline", *map(fmt4, mnb)])

    # inner dev for meta on NTU train
    idx2 = np.arange(len(trn_text))
    tr2_idx, dev2_idx = train_test_split(idx2, test_size=0.25, random_state=seed, stratify=trn_y)

    tr2_text = [trn_text[i] for i in tr2_idx]
    tr2_y = trn_y[tr2_idx]
    dev2_text = [trn_text[i] for i in dev2_idx]
    dev2_y = trn_y[dev2_idx]

    X2_tr, X2_dev = build_text_X(tr2_text, dev2_text)
    text2 = train_text_clf(X2_tr, tr2_y, seed)
    logits2_dev = text2.decision_function(X2_dev)
    onto2_dev = build_ontology_features(dev2_text, topics_owl=None).toarray()

    C2, thr2 = tune_meta(logits2_dev, onto2_dev, dev2_y, seed)

    meta2 = LogisticRegression(max_iter=500, multi_class="multinomial", solver="lbfgs", C=C2)
    meta2.fit(np.hstack([logits2_dev, onto2_dev]), dev2_y)

    logits_te = text_n.decision_function(Xn_te)
    onto_te = build_ontology_features(ten_text, topics_owl=None).toarray()
    pred_n_onto = meta_override_predict(meta2, logits_te, onto_te, pred_n_base, thr2)

    mno = macro_metrics(ten_y, pred_n_onto)
    rows.append([seed, "NTU survey (silver labels)", "Ontology-assisted (ours)", *map(fmt4, mno)])


df5a = pd.DataFrame(rows, columns=[
    "Seed", "Dataset", "Model",
    "Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"
])
df5a.to_csv(TABLE5A_OUT, index=False)


# ----------------------------
# 7) Table 5B mean ± std
# ----------------------------
metrics = ["Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"]
grp = df5a.groupby(["Dataset", "Model"])[metrics]
mean_df = grp.mean().reset_index()
std_df = grp.std(ddof=1).reset_index()
table5b = mean_df.merge(std_df, on=["Dataset", "Model"], suffixes=("_mean", "_std"))

rows_b = []
for _, r in table5b.iterrows():
    out = {"Dataset": r["Dataset"], "Model": r["Model"]}
    for m in metrics:
        out[f"{m} (mean ± std)"] = f"{r[m+'_mean']:.4f} ± {r[m+'_std']:.4f}"
    rows_b.append(out)

df5b = pd.DataFrame(rows_b)
df5b.to_csv(TABLE5B_OUT, index=False, encoding="utf-8-sig")

print("Saved:", OWL_OUT, NTU_OUT_CSV, TABLE5A_OUT, TABLE5B_OUT)
print(df5b)