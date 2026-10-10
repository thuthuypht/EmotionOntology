"""
run_all.py  –  Chạy toàn bộ pipeline từ đầu đến cuối
=====================================================================
Thứ tự:
  Bước 0 → src/00_chuan_hoa_dulieu.py   → data/processed/ + data/labels/
  Bước 1 → src/01_mo_hinh_baseline.py   → results/Ket_Qua_Baseline.xlsx
  Bước 2 → src/02_mo_hinh_phobert.py    → results/Ket_Qua_So_Sanh_...xlsx
  Bước 3 → src/03_ontology_builder.py   → ontology/Ontology_...V7.owl
  Bước 4 → src/04_sparql_queries.py     → results/Ket_Qua_SPARQL.* + ontology/SPARQL_Queries.txt

Sau đó chạy dashboard:
  python -m streamlit run dashboard/05_dashboard.py
=====================================================================
"""

import subprocess, sys, os, time
from config import PATHS

ROOT   = os.path.dirname(os.path.abspath(__file__))
SRC    = os.path.join(ROOT, "src")

STEPS = [
    ("00_chuan_hoa_dulieu.py",           "Chuẩn hóa dữ liệu (rule-based V7)"),
    ("06_data_enrichment.py",            "Làm giàu dữ liệu lớp thiểu số (V8)"),
    ("01_mo_hinh_baseline.py",           "Mô hình Baseline (TF-IDF + ML) ← KẾT QUẢ THỰC TẾ"),
    ("05_inter_annotator_agreement.py",  "Inter-Annotator Agreement + Cohen's Kappa"),
    ("03_ontology_builder.py",           "Xây dựng Ontology OWL V8 (dữ liệu đã làm giàu)"),
    ("04_sparql_queries.py",             "SPARQL Queries (pre-computed)"),
    # PhoBERT: chỉ chạy khi có GPU – hướng mở rộng, không phải kết quả chính
    # ("02_mo_hinh_phobert.py",  "PhoBERT Fine-tuning (cần GPU – chạy trên Colab)"),
]

def run_step(script: str, name: str) -> bool:
    path = os.path.join(SRC, script)
    print(f"\n{'═'*60}")
    print(f"  ▶  {name}")
    print(f"{'═'*60}")
    t0     = time.time()
    result = subprocess.run([sys.executable, path])
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"  {'✓' if ok else '✗'}  {'OK' if ok else 'Lỗi!'} ({elapsed:.1f}s)")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("  NTU FEEDBACK ANALYSIS SYSTEM — PIPELINE RUNNER")
    print("=" * 60)

    all_ok = True
    for script, name in STEPS:
        all_ok = run_step(script, name) and all_ok

    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅  TẤT CẢ BƯỚC HOÀN TẤT!")
        print()
        print("  📁 Files đầu ra:")
        iaa_path = os.path.join(os.path.dirname(PATHS["gold_set"]), "IAA_200_Results.xlsx")
        checks = [
            ("data_final [THUC TE]",    PATHS["data_final"]),
            ("gold_set [THUC TE]",      PATHS["gold_set"]),
            ("IAA_Results [THUC TE]",   iaa_path),
            ("label_rules",             PATHS["label_rules"]),
            ("result_baseline [THUC TE]", PATHS["result_baseline"]),
            ("result_compare [THUC TE]",PATHS["result_compare"]),  # PhoBERT removed from main results
            ("ontology_owl [THUC TE]",  PATHS["ontology_owl"]),
            ("result_sparql_xlsx",      PATHS["result_sparql_xlsx"]),
            ("sparql_txt",              PATHS["sparql_txt"]),
        ]
        for key, path in checks:
            exists  = "✓" if os.path.exists(path) else "✗"
            size_kb = os.path.getsize(path)//1024 if os.path.exists(path) else 0
            rel     = os.path.relpath(path, ROOT)
            print(f"    {exists}  {rel:<50}  ({size_kb} KB)")

        print()
        print("  🚀  Khởi động dashboard:")
        print("      python -m streamlit run dashboard/05_dashboard.py")
    else:
        print("  ⚠️   Một số bước gặp lỗi — kiểm tra output ở trên.")
    print("=" * 60)
