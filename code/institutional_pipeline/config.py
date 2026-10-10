"""
config.py  –  Cấu hình đường dẫn tập trung cho toàn bộ đồ án
=====================================================================
Tất cả các script đều import từ đây thay vì hardcode tên file.
Cách dùng trong script con:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import PATHS
=====================================================================
"""

import os

# ── Thư mục gốc của dự án (nơi chứa config.py này) ──────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Các thư mục con ─────────────────────────────────────────────────
DIR = {
    "data_raw"       : os.path.join(ROOT, "data", "raw"),
    "data_processed" : os.path.join(ROOT, "data", "processed"),
    "data_labels"    : os.path.join(ROOT, "data", "labels"),
    "results"        : os.path.join(ROOT, "results"),
    "ontology"       : os.path.join(ROOT, "ontology"),
    "src"            : os.path.join(ROOT, "src"),
    "dashboard"      : os.path.join(ROOT, "dashboard"),
}

# Tự động tạo thư mục nếu chưa có
for _d in DIR.values():
    os.makedirs(_d, exist_ok=True)

# ── Đường dẫn file đầy đủ ────────────────────────────────────────────
PATHS = {
    # ── Dữ liệu đầu vào ──────────────────────────────────────────────
    "raw_data"          : os.path.join(DIR["data_raw"],
                                        "Khoa CNTT - Gop y.xlsx"),

    # ── Dữ liệu đã xử lý ────────────────────────────────────────────
    "data_ntu_official": os.path.join(DIR["data_processed"], "Du_Lieu_NTU_Official.xlsx"),
    "data_train_full"  : os.path.join(DIR["data_processed"], "Du_Lieu_Chuan_Final.xlsx"),
    "data_final"        : os.path.join(DIR["data_processed"],
                                        "Du_Lieu_Chuan_Final.xlsx"),
    "data_human"        : os.path.join(DIR["data_processed"],
                                        "Du_Lieu_Chuan_Human.xlsx"),
    "iaa_annotations": os.path.join(DIR["data_processed"], "IAA_Human2_Annotations.xlsx"),
    "gold_set"          : os.path.join(DIR["data_processed"],
                                        "Gold_Set_1000.xlsx"),

    # ── Tài liệu nhãn ───────────────────────────────────────────────
    "label_rules"       : os.path.join(DIR["data_labels"],
                                        "Quy_Tac_Gan_Nhan.txt"),

    # ── Kết quả thực nghiệm ──────────────────────────────────────────
    "result_baseline"   : os.path.join(DIR["results"],
                                        "Ket_Qua_Baseline.xlsx"),
    "result_compare"    : os.path.join(DIR["results"],
                                        "Ket_Qua_So_Sanh_3_Phuong_Phap.xlsx"),
    "result_sparql_xlsx": os.path.join(DIR["results"],
                                        "Ket_Qua_SPARQL.xlsx"),
    "result_sparql_json": os.path.join(DIR["results"],
                                        "Ket_Qua_SPARQL.json"),

    # ── Ontology ─────────────────────────────────────────────────────
    "ontology_owl"      : os.path.join(DIR["ontology"],
                                        "Ontology_NTU_FeedbackSystem_V8.owl"),
    "sparql_txt"        : os.path.join(DIR["ontology"],
                                        "SPARQL_Queries.txt"),

    # ── Mô hình PhoBERT (thư mục lưu model sau fine-tune) ────────────
    "phobert_sentiment" : os.path.join(ROOT, "models", "phobert_sentiment"),
    "phobert_topic"     : os.path.join(ROOT, "models", "phobert_topic"),
    "phobert_emotion"   : os.path.join(ROOT, "models", "phobert_emotion"),
}


def show():
    """In tóm tắt tất cả đường dẫn (dùng để debug)."""
    print(f"\n{'─'*60}")
    print(f"  ROOT: {ROOT}")
    print(f"{'─'*60}")
    for name, path in PATHS.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists}  {name:<22} {os.path.relpath(path, ROOT)}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    show()
