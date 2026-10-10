"""
=============================================================================
BƯỚC 1: MÔ HÌNH NLP BASELINE - TF-IDF + ML Classifiers
=============================================================================
Mục tiêu: Huấn luyện và đánh giá 3 mô hình phân loại:
  - Logistic Regression (LR)
  - Naive Bayes (NB)
  - Linear SVM (SVM)
Cho 3 bài toán: Sentiment, Topic, Emotion classification

Output: Ket_Qua_Baseline.xlsx, bảng Precision/Recall/F1, Confusion Matrix
=============================================================================
"""


import sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from config import PATHS

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder
import re

print("=" * 65)
print("BƯỚC 1: MÔ HÌNH BASELINE  (TF-IDF + LR / NB / SVM)")
print("=" * 65)

# ============================================================
# 1. ĐỌC DỮ LIỆU GOLD SET (nhãn thủ công - cột Human)
#    Nếu chưa có gold set riêng, dùng Du_Lieu_Chuan_Final.xlsx
# ============================================================
try:
    df = pd.read_excel(PATHS["data_human"])
    # Map lại tên cột về chuẩn chung
    col_map = {
        "NoiDung": "noi_dung", "Stt": "stt",
        "Sentiment": "sentiment", "Emotion": "emotion",
        "Topic": "topic", "CauHoi": "cau_hoi",
        "Lecturer": "target"
    }
    df.rename(columns=col_map, inplace=True)
    print(f"✓ Dùng file nhãn thủ công: Du_Lieu_Chuan_Human.xlsx ({len(df)} dòng)")
except FileNotFoundError:
    # ── Baseline ML: dùng toàn bộ dữ liệu gồm cả augmented ─────────────────
    # (HAND_LABELED_V8, TEMPLATE_AUG_V8 chỉ dùng để train, không đưa vào Ontology)
    df = pd.read_excel(PATHS["data_train_full"])
    print(f"✓ Dùng file nhãn tự động: Du_Lieu_Chuan_Final.xlsx ({len(df)} dòng)")

df = df.dropna(subset=["noi_dung"]).copy()
df["noi_dung"] = df["noi_dung"].astype(str)

# ============================================================
# 2. TIỀN XỬ LÝ VĂN BẢN (Vietnamase preprocessing)
# ============================================================
STOP_WORDS = {
    "và", "của", "là", "có", "cho", "không", "được", "trong", "này", "với",
    "một", "các", "về", "tôi", "em", "thầy", "cô", "giáo", "viên", "học",
    "sinh", "ạ", "ơi", "nha", "nhé", "thôi", "lắm", "quá", "rất", "hơi",
    "đã", "sẽ", "đang", "vẫn", "còn", "cũng", "như", "mà", "nên", "hay",
    "hoặc", "vì", "nếu", "thì", "bị", "do", "từ", "đến", "theo", "lên",
    "ra", "vào", "lại", "đi", "đây", "đó", "khi", "sau", "trước", "hết"
}

def preprocess(text: str) -> str:
    text = str(text).lower().strip()
    # Xóa ký tự đặc biệt, giữ chữ và khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)

df["text_clean"] = df["noi_dung"].apply(preprocess)

# ============================================================
# 3. ĐỊNH NGHĨA MÔ HÌNH
# ============================================================
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0,
                                               class_weight="balanced",
                                               random_state=42),
    "Naive Bayes"        : MultinomialNB(alpha=0.5),
    "Linear SVM"         : LinearSVC(max_iter=2000, C=1.0,
                                      class_weight="balanced",
                                      random_state=42),
}

TFIDF = TfidfVectorizer(
    ngram_range=(1, 2),   # Unigram + Bigram
    max_features=10000,
    min_df=2,
    sublinear_tf=True,
)

# ============================================================
# 4. HÀM ĐÁNH GIÁ TỔNG HỢP (5-fold Cross Validation)
# ============================================================
def evaluate_task(task_name: str, y_series: pd.Series, label_order=None):
    """
    Đánh giá toàn bộ 3 mô hình cho một bài toán phân loại.
    Dùng Stratified 5-Fold Cross Validation.
    Xuất: classification report, Confusion Matrix, nhận xét lớp thiểu số.
    """
    print(f"\n{'─'*65}")
    print(f"  BÀI TOÁN: {task_name.upper()}")
    print(f"{'─'*65}")

    y = y_series.copy()
    counts = y.value_counts()
    valid_labels = counts[counts >= 5].index
    mask = y.isin(valid_labels)
    X_text = df.loc[mask, "text_clean"]
    y = y[mask]

    X = TFIDF.fit_transform(X_text)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_results = []
    cm_results  = []   # Lưu confusion matrices

    for name, model in MODELS.items():
        y_pred = cross_val_predict(model, X, y, cv=cv)
        acc    = accuracy_score(y, y_pred)
        mf1    = f1_score(y, y_pred, average="macro", zero_division=0)

        print(f"\n  [{name}]")
        print(f"  Chia dữ liệu: Stratified 5-Fold Cross Validation (random_state=42)")
        print(f"  Tổng mẫu    : {len(y)}  |  Số lớp hợp lệ: {y.nunique()}")
        print(f"  Accuracy    : {acc*100:.2f}%")
        print(f"  Macro-F1    : {mf1*100:.2f}%")
        print()
        print(classification_report(y, y_pred, zero_division=0, digits=3))

        # ── Confusion Matrix ──────────────────────────────────────────────
        labels = sorted(y.unique())
        cm = confusion_matrix(y, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"Thực_{l}" for l in labels],
                                  columns=[f"Đoán_{l}" for l in labels])
        print(f"  Confusion Matrix (hàng=Thực tế, cột=Dự đoán):")
        print(cm_df.to_string())

        # ── Nhận xét lớp thiểu số ────────────────────────────────────────
        minority_labels = counts[counts < counts.max() * 0.05].index.tolist()
        minority_labels = [l for l in minority_labels if l in labels]
        if minority_labels:
            print(f"\n  ⚠ Lớp thiểu số: {minority_labels}")
            for ml in minority_labels:
                idx = labels.index(ml)
                n_correct = cm[idx, idx]
                n_total   = cm[idx].sum()
                print(f"    {ml}: {n_correct}/{n_total} đúng "
                      f"({n_correct/n_total*100:.1f}% Recall) "
                      f"– mất cân bằng lớp ảnh hưởng đến Macro-F1")

        # ── Thu thập kết quả ──────────────────────────────────────────────
        report = classification_report(y, y_pred, zero_division=0,
                                        output_dict=True)
        for lbl, metrics in report.items():
            if isinstance(metrics, dict):
                all_results.append({
                    "task"      : task_name,
                    "model"     : name,
                    "label"     : lbl,
                    "precision" : round(metrics["precision"], 4),
                    "recall"    : round(metrics["recall"], 4),
                    "f1-score"  : round(metrics["f1-score"], 4),
                    "support"   : int(metrics["support"]),
                    "accuracy"  : round(acc, 4),
                    "macro_f1"  : round(mf1, 4),
                })

        # ── Lưu confusion matrix ──────────────────────────────────────────
        for i, true_lbl in enumerate(labels):
            for j, pred_lbl in enumerate(labels):
                cm_results.append({
                    "task"      : task_name,
                    "model"     : name,
                    "true_label": true_lbl,
                    "pred_label": pred_lbl,
                    "count"     : int(cm[i, j]),
                    "is_correct": true_lbl == pred_lbl,
                })

    return all_results, cm_results

# ============================================================
# 5. CHẠY ĐÁNH GIÁ CHO 3 BÀI TOÁN
# ============================================================
all_results = []
all_cm      = []

r, cm = evaluate_task("Sentiment", df["sentiment"],
                       label_order=["Positive", "Neutral", "Negative"])
all_results += r;  all_cm += cm

r, cm = evaluate_task("Topic", df["topic"],
                       label_order=["GiangDay","TaiLieu","CoSoVatChat",
                                     "ThoiGian","DanhGiaCongBang","ThaiDo","Khac"])
all_results += r;  all_cm += cm

r, cm = evaluate_task("Emotion", df["emotion"],
                       label_order=["KhenNgoi","HaiLong","DeXuat",
                                     "PhanNan","ThatVong","KhongYKien"])
all_results += r;  all_cm += cm

# ============================================================
# 6. SO SÁNH TỔNG HỢP 3 MÔ HÌNH
# ============================================================
print("\n" + "=" * 65)
print("  TÓM TẮT SO SÁNH (Accuracy & Macro-F1 - 5-Fold CV)")
print("=" * 65)

# (summary pivot exportado na seção 7)

# ============================================================
# 7. XUẤT KẾT QUẢ RA EXCEL
# ============================================================
df_res  = pd.DataFrame(all_results)
df_cm   = pd.DataFrame(all_cm)

# Summary pivot
summary_df = (df_res[df_res["label"] == "macro avg"]
              .drop(columns=["label","precision","recall","f1-score","support"])
              .drop_duplicates(subset=["task","model"]))
pivot = summary_df.pivot_table(index="model", columns="task",
                                values=["accuracy","macro_f1"]).round(4)

with pd.ExcelWriter(PATHS["result_baseline"], engine="openpyxl") as writer:
    df_res.to_excel(writer, sheet_name="Classification_Report", index=False)
    df_cm.to_excel(writer,  sheet_name="Confusion_Matrix",       index=False)
    pivot.to_excel(writer,  sheet_name="Summary_Pivot")

print(f"\n✓ Đã lưu: Ket_Qua_Baseline.xlsx  (3 sheets)")
print(f"  Sheet 1 – Classification_Report : {len(df_res)} dòng")
print(f"  Sheet 2 – Confusion_Matrix       : {len(df_cm)} dòng")
print(f"  Sheet 3 – Summary_Pivot          : tóm tắt 3 mô hình")

# ============================================================
# 8. LƯU MÔ HÌNH TỐT NHẤT (Rule-based vs Best ML vs PhoBERT)
# ============================================================
print("\n" + "=" * 65)
print("  SẴN SÀNG CHO SO SÁNH 3 PHƯƠNG PHÁP:")
print("  1. Rule-based  (accuracy.py gốc)")
print("  2. Baseline ML (file này)")
print("  3. PhoBERT     (02_mo_hinh_phobert.py)")
print("=" * 65)
print("\n[HOÀN TẤT BƯỚC 1]")
