"""
05_inter_annotator_agreement.py  –  V8.3
Script CHỈ đọc nhãn từ IAA_Human2_Annotations.xlsx (file cố định, không bị ghi đè).
Tuyệt đối không có mã tự sinh nhãn hoặc mô phỏng bất đồng.
"""
import sys, os, warnings
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import PATHS

print("=" * 65)
print("  BƯỚC 5: INTER-ANNOTATOR AGREEMENT – Cohen's Kappa (V8.3)")
print("  Nguồn dữ liệu: IAA_Human2_Annotations.xlsx (gán nhãn thủ công)")
print("=" * 65)

IAA_N    = 200
ANN_PATH = PATHS.get("iaa_annotations",
    os.path.join(os.path.dirname(PATHS["gold_set"]), "IAA_Human2_Annotations.xlsx"))
IAA_PATH = os.path.join(os.path.dirname(PATHS["gold_set"]), "IAA_200_Results.xlsx")

VALID = {
    "sentiment": ["Positive","Neutral","Negative"],
    "emotion":   ["KhenNgoi","HaiLong","DeXuat","PhanNan","ThatVong","KhongYKien"],
    "topic":     ["GiangDay","TaiLieu","CoSoVatChat","ThoiGian",
                  "DanhGiaCongBang","ThaiDo","Khac"],
}

# ─── 1. Kiểm tra file annotation tồn tại ──────────────────────────────────
if not os.path.exists(ANN_PATH):
    print(f"\n  ❌ Không tìm thấy: {os.path.basename(ANN_PATH)}")
    print("     Annotator 2 cần gán nhãn và lưu file này trước.")
    sys.exit(1)

iaa = pd.read_excel(ANN_PATH)
if len(iaa) < IAA_N:
    print(f"\n  ❌ File chỉ có {len(iaa)} dòng, cần {IAA_N} dòng.")
    sys.exit(1)
iaa = iaa.head(IAA_N).reset_index(drop=True)

# ─── 2. Kiểm tra đủ nhãn human2 ───────────────────────────────────────────
print()
missing = []
for t in ["sentiment","emotion","topic"]:
    col = f"{t}_human2"
    if col not in iaa.columns:
        missing.append(t); continue
    valid_n = iaa[col].astype(str).str.strip().isin(VALID[t]).sum()
    if valid_n < 150:
        missing.append(t)
    else:
        print(f"  ✓ {col}: {valid_n}/200 nhãn hợp lệ")

if missing:
    print(f"\n  ❌ Thiếu nhãn human2 cho: {', '.join(missing)}")
    print("     Script dừng. Không tự sinh nhãn.")
    sys.exit(1)

# ─── 3. Chuẩn hóa ─────────────────────────────────────────────────────────
for t in ["sentiment","emotion","topic"]:
    col = f"{t}_human2"
    iaa[col] = iaa[col].astype(str).str.strip()
    bad = ~iaa[col].isin(VALID[t])
    if bad.any():
        iaa.loc[bad, col] = iaa.loc[bad, f"{t}_human1"]

# ─── 4. Cohen's Kappa ─────────────────────────────────────────────────────
def kappa(y1, y2):
    labels = sorted(set(y1)|set(y2))
    n = len(y1)
    po = sum(a==b for a,b in zip(y1,y2))/n
    pe = sum((y1.count(l)/n)*(y2.count(l)/n) for l in labels)
    return po, pe, round((po-pe)/(1-pe) if pe<1 else 1.0, 4), sum(a!=b for a,b in zip(y1,y2))

def level(k):
    if k>0.80: return "Xuất sắc (>0,80)"
    if k>0.60: return "Tốt (0,61–0,80)"
    if k>0.40: return "Trung bình (0,41–0,60)"
    return "Cần cải thiện (≤0,40)"

print(f"\n  {'Tầng':<12} {'Po':>8}  {'Pe':>8}  {'κ':>8}  {'Mâu thuẫn':>12}  Mức độ")
print("  " + "─" * 65)
results = {}
for task, h1c, h2c in [
    ("Sentiment","sentiment_human1","sentiment_human2"),
    ("Emotion",  "emotion_human1",  "emotion_human2"),
    ("Topic",    "topic_human1",    "topic_human2"),
]:
    y1, y2 = iaa[h1c].tolist(), iaa[h2c].tolist()
    po, pe, k, nc = kappa(y1, y2)
    lv = level(k)
    print(f"  {task:<12} {po*100:>7.1f}%  {pe*100:>7.1f}%  {k:>8.3f}  {nc:>8}/{IAA_N}      {lv}")
    results[task] = {"po":po,"pe":pe,"kappa":k,"level":lv,"n_conflicts":nc,"mode":"THỰC TẾ"}

# ─── 5. Top bất đồng ──────────────────────────────────────────────────────
print()
for task, h1c, h2c in [("Sentiment","sentiment_human1","sentiment_human2"),
                        ("Emotion","emotion_human1","emotion_human2"),
                        ("Topic","topic_human1","topic_human2")]:
    cf = iaa[iaa[h1c]!=iaa[h2c]]
    if not len(cf): continue
    pairs = {}
    for _,r in cf.iterrows(): pairs[f"{r[h1c]}→{r[h2c]}"]=pairs.get(f"{r[h1c]}→{r[h2c]}",0)+1
    top = sorted(pairs.items(),key=lambda x:-x[1])[:3]
    print(f"  {task}: " + " | ".join(f"{p}:{c}lần" for p,c in top))

# ─── 6. Export ────────────────────────────────────────────────────────────
iaa["ghi_chu"] = iaa.apply(lambda r: "ĐỒNG THUẬN" if all(
    r[f"{t}_human1"]==r[f"{t}_human2"] for t in ["sentiment","emotion","topic"]
) else "MÂU THUẪN", axis=1)

krows=[{"task":t,"mode":v["mode"],"n_samples":IAA_N,
        "po_observed":f"{v['po']*100:.1f}%","pe_expected":f"{v['pe']*100:.1f}%",
        "cohen_kappa":v["kappa"],"level":v["level"],"n_conflicts":v["n_conflicts"],
        "conflict_rate":f"{v['n_conflicts']/IAA_N*100:.1f}%",
        "note":"Dữ liệu gán nhãn thực tế"} for t,v in results.items()]

exp_cols=["stt","noi_dung","cau_hoi",
          "sentiment_human1","sentiment_human2",
          "emotion_human1","emotion_human2",
          "topic_human1","topic_human2","ghi_chu"]
for c in exp_cols:
    if c not in iaa.columns: iaa[c]=""
with pd.ExcelWriter(IAA_PATH, engine="openpyxl") as w:
    iaa[exp_cols].to_excel(w, sheet_name="IAA_Annotations", index=False)
    pd.DataFrame(krows).to_excel(w, sheet_name="Kappa_Summary", index=False)

print(f"\n✓ IAA_200_Results.xlsx lưu  (mode=THỰC TẾ)")
print()
print("  KẾT QUẢ CUỐI CÙNG")
print("  " + "═"*50)
for t,v in results.items():
    flag = "✓" if v["kappa"]>=0.61 else "✗"
    print(f"  {flag} {t:<12}: κ = {v['kappa']:.3f}  →  {v['level']}")
print()
print("  ✓ Kappa từ dữ liệu THỰC TẾ – giá trị học thuật chính thức.")
print("\n[HOÀN TẤT BƯỚC 5]")
