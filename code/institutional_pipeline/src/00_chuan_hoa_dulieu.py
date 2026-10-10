"""
=============================================================================
BƯỚC 0: CHUẨN HÓA DỮ LIỆU - Tạo file dữ liệu chuẩn thống nhất
=============================================================================
Output:
  - Du_Lieu_Chuan_Final.xlsx   : Toàn bộ dữ liệu đã gán nhãn rule-based
  - Gold_Set_1000.xlsx         : 1000 dòng đầu để gán nhãn thủ công
  - Quy_Tac_Gan_Nhan.txt       : Quy tắc gán nhãn rõ ràng
=============================================================================
"""


import sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from config import PATHS

import pandas as pd
import openpyxl
import re
import os

print("=" * 60)
print("BƯỚC 0: CHUẨN HÓA DỮ LIỆU")
print("=" * 60)

# ============================================================
# 1. ĐỌC VÀ PARSE DỮ LIỆU GỐC
# ============================================================
RAW_FILE = PATHS["raw_data"]
wb = openpyxl.load_workbook(RAW_FILE)
ws = wb.active

# Header thật sự ở dòng 8 (index 7), dữ liệu từ dòng 9 (index 8)
data_rows = []
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i < 8:
        continue   # Bỏ metadata header
    if row[0] is None:
        continue   # Bỏ dòng trống
    stt, mon_hoc, nhom_hp, can_bo, cau_hoi, noi_dung = row[:6]
    if noi_dung is None or str(noi_dung).strip() == "":
        continue   # Bỏ dòng không có nội dung

    # Tách mã học phần và tên học phần
    mon_str = str(mon_hoc or "").strip()
    if " - " in mon_str:
        parts = mon_str.split(" - ", 1)
        hoc_phan = parts[0].strip()
    else:
        hoc_phan = mon_str

    # Tách mã GV
    gv_str = str(can_bo or "").strip()
    ma_gv = gv_str.split(" - ")[0].strip() if " - " in gv_str else gv_str

    data_rows.append({
        "stt": stt,
        "hoc_phan": hoc_phan,
        "nhom_lop": str(nhom_hp or "").strip(),
        "ma_gv": ma_gv,
        "cau_hoi": str(cau_hoi or "").strip(),
        "noi_dung": str(noi_dung or "").strip(),
    })

df = pd.DataFrame(data_rows)
df["stt"] = range(1, len(df) + 1)   # Đánh số lại liên tục
print(f"✓ Đọc được {len(df)} dòng dữ liệu hợp lệ từ file gốc.")

# ============================================================
# 2. ĐỊNH NGHĨA HỆ NHÃN CHUẨN (dùng thống nhất toàn đồ án)
# ============================================================
# Sentiment: Positive / Neutral / Negative
# Emotion  : KhenNgoi / HaiLong / DeXuat / PhanNan / ThatVong / KhongYKien
# Topic    : GiangDay / TaiLieu / CoSoVatChat / ThoiGian / DanhGiaCongBang / ThaiDo / Khac
# Target   : GV_<ma_gv> hoặc NhaTruong

TOPIC_KW = {
    # ── Thứ tự ưu tiên trong dict (Python duyệt theo thứ tự khai báo) ────────
    # ThaiDo TRƯỚC GiangDay để tránh gán nhầm (lỗi G3 – phân tích V7)
    "ThaiDo": [
        "thái độ",            # keyword rõ ràng nhất
        "lịch sự",
        "tôn trọng sinh viên",
        "khó chịu với sv",
        "giao tiếp với sinh viên",
        "ứng xử với sv",
    ],
    # TaiLieu TRƯỚC GiangDay (lỗi G4 – phân tích V7)
    "TaiLieu": [
        "slide", "giáo trình", "bài tập",
        "tài liệu học", "lms", "e-learning",
        "sách giáo khoa", "tài nguyên học",
        "bài giảng điện tử", "upload tài liệu",
    ],
    "DanhGiaCongBang": [
        "chấm điểm", "kết quả học tập", "thi cử", "công bằng trong đánh giá",
        "điểm danh", "tính điểm", "điểm số", "điểm thi", "điểm kiểm tra",
        "bài kiểm tra", "bài thi", "kết quả thi",
    ],
    "ThoiGian": [
        "đi trễ", "đến trễ", "vào trễ", "đúng giờ", "không đúng giờ",
        "nghỉ dạy", "bù giờ", "ra chơi", "dạy nhanh", "dạy chậm",
        "muộn giờ", "trễ giờ", "đổi lịch", "thông báo trễ",
    ],
    "ThaiDo": [
        "thái độ", "giao tiếp với sinh viên", "lịch sự", "tôn trọng sinh viên",
        "khó chịu", "kiêu ngạo", "cởi mở", "hoà đồng với sinh viên",
        "thân thiện với sinh viên",
    ],
    "GiangDay": [
        "giảng dạy", "phương pháp dạy", "truyền đạt kiến thức",
        "dễ hiểu", "khó hiểu", "nhiệt tình", "tận tâm", "giảng bài",
        "dạy học", "có tâm", "nội dung môn", "lý thuyết", "ví dụ thực tế",
        "thực hành",           # "thực hành" → GiangDay (không phải CoSoVatChat)
        "tiết thực hành",
        "bài thực hành",
        "buổi thực hành",
        "nên kết hợp thực hành",
        "thêm thực hành",
        "nhiều thực hành hơn",
        "dạy thực hành",
    ],
    "TaiLieu": [
        "slide", "giáo trình", "bài tập", "tài liệu học",
        "lms", "e-learning", "sách giáo khoa", "tài nguyên học",
        "bài giảng điện tử", "upload tài liệu",
    ],
    "CoSoVatChat": [
        "wifi", "mạng internet", "máy chiếu", "phòng học", "bàn ghế",
        "điều hòa", "phòng nóng", "quạt", "âm thanh", "micro",
        "loa", "nhà xe", "giữ xe", "cơ sở vật chất",
        "phòng máy", "máy tính", "thiết bị",
    ],
}

WISH_WORDS   = ["mong", "nên cho", "cần thêm", "hy vọng", "ước",
                 "đề nghị", "giá như", "phải chi",
                 "điều chỉnh", "thay đổi", "nâng cao", "cải thiện",
                 "bổ sung", "tăng cường", "giảm bớt"]
# Tách 'nên' ra khỏi WISH_WORDS vì hay gây nhầm ('nên kết hợp' → GiangDay)
WISH_WORD_STANDALONE = ["mong", "hy vọng", "ước", "đề nghị", "giá như",
                         "phải chi"]

NEG_PHRASES  = ["không hay", "không tốt", "không hiểu", "không nhiệt tình",
                 "chưa tốt", "chưa hay", "không rõ", "dạy chán", "khó hiểu",
                 "hơi nhanh", "nói nhỏ", "nói khó nghe", "lòng vòng",
                 "thiếu rõ", "yếu kém", "không ổn", "không hài lòng",
                 "không tâm", "không quan tâm", "dạy qua loa"]
PRAISE_WORDS = ["tuyệt vời", "xuất sắc", "rất hay", "rất tốt", "giỏi",
                 "nhiệt tình", "tận tâm", "dễ hiểu", "hài lòng",
                 "thân thiện", "vui vẻ", "tích cực", "chu đáo", "quan tâm",
                 "chi tiết", "dễ thương", "cute", "có tâm",
                 "vui tính", "hòa đồng", "chịu khó", "dễ gần",
                 "rất tốt", "quá tốt", "hoàn hảo"]
DISAPPOINT   = ["thất vọng", "quá tệ", "không thể chấp nhận",
                 "rất tệ", "cực tệ", "tệ hại", "quá kém", "vô trách nhiệm",
                 "chán nản thật sự"]
EMPTY_KW     = ["không", "không có", "chưa có", "nothing", "none", "..."]


def classify(text: str, cau_hoi: str):
    t = str(text).lower().strip()
    cq = str(cau_hoi).lower()

    # ── TOPIC ──────────────────────────────────────────────────────────────
    # Kiểm tra từng nhóm theo thứ tự ưu tiên trong dict
    topic = "Khac"
    for tp, kws in TOPIC_KW.items():
        if any(k in t for k in kws):
            topic = tp
            break

    # ── Xử lý câu hỏi "Nhà trường" (sửa V7 – dựa trên phân tích lỗi G1+G2) ──
    # Ưu tiên 1: keyword GV/dạy học → GiangDay  (lỗi G1)
    # Ưu tiên 2: keyword CSVC → CoSoVatChat     (lỗi G2)
    # Mặc định: → Khac (không gán CoSoVatChat chung chung)
    NT_GIANG_DAY_KW = [
        "giáo viên", "giảng viên", "thầy ", "cô ", "dạy ", "giảng ",
        "học phần", "môn học", "chương trình học", "phương pháp",
        "nội dung học", "hướng dẫn", "tăng lương", "thêm giáo viên",
    ]
    CSVC_KEYWORDS = [
        "wifi", "mạng", "máy chiếu", "phòng học", "bàn ghế",
        "điều hòa", "nóng", "quạt", "âm thanh", "micro", "loa",
        "nhà xe", "giữ xe", "cơ sở vật chất",
        "phòng máy", "máy tính", "thiết bị", "hệ thống máy",
        "mở cửa sớm", "ký túc xá", "căng tin", "thư viện",
        "phòng công tác", "nhân sự phòng",
    ]
    if "nhà trường" in cq and topic == "Khac":
        if any(k in t for k in NT_GIANG_DAY_KW):
            topic = "GiangDay"          # ưu tiên 1
        elif any(k in t for k in CSVC_KEYWORDS):
            topic = "CoSoVatChat"        # ưu tiên 2
        # else: giữ Khac

    # ── Ví dụ biên: 'thực hành' + 'máy chiếu' → GiangDay thắng CoSoVatChat
    # (đã được xử lý bằng cách đặt GiangDay với 'thực hành' trước CoSoVatChat trong dict)

    # ── SENTIMENT & EMOTION ───────────────────────────────────────────────
    # Ưu tiên 1: Quá ngắn hoặc không ý kiến
    if len(t) < 5 or t in EMPTY_KW:
        return "Neutral", "KhongYKien", topic

    # Ưu tiên 2: Thất vọng nặng
    if any(p in t for p in DISAPPOINT):
        return "Negative", "ThatVong", topic

    # Ưu tiên 3: Câu đề xuất rõ ràng
    is_wish = (
        any(w in t for w in WISH_WORD_STANDALONE)
        or any(w in t for w in WISH_WORDS)
    )
    if is_wish:
        return "Neutral", "DeXuat", topic

    # Ưu tiên 4: Câu phủ định / chê rõ
    if any(p in t for p in NEG_PHRASES):
        return "Negative", "PhanNan", topic

    # Ưu tiên 5: Khen
    if any(w in t for w in PRAISE_WORDS):
        return "Positive", "KhenNgoi", topic

    # Ưu tiên 6: Dựa vào loại câu hỏi
    if "ưu điểm" in cq:
        return "Positive", "HaiLong", topic
    if "góp ý cho gv" in cq or "góp ý cho nhà trường" in cq:
        return "Neutral", "DeXuat", topic

    return "Neutral", "KhongYKien", topic


def get_target(ma_gv: str, cau_hoi: str):
    if "nhà trường" in str(cau_hoi).lower():
        return "NhaTruong"
    return f"GV_{ma_gv}" if ma_gv else "KhongXacDinh"


print("→ Đang gán nhãn tự động (rule-based baseline)...")
labels = df.apply(lambda r: classify(r["noi_dung"], r["cau_hoi"]), axis=1)
df["sentiment"] = [x[0] for x in labels]
df["emotion"]   = [x[1] for x in labels]
df["topic"]     = [x[2] for x in labels]
df["target"]    = df.apply(lambda r: get_target(r["ma_gv"], r["cau_hoi"]), axis=1)

# ============================================================
# 3. THỐNG KÊ PHÂN PHỐI NHÃN
# ============================================================
print("\n── PHÂN PHỐI NHÃN ──")
for col in ["sentiment", "emotion", "topic"]:
    print(f"\n{col.upper()}:")
    print(df[col].value_counts().to_string())

# ============================================================
# 4. XUẤT FILE CHUẨN
# ============================================================
COLS = ["stt", "hoc_phan", "nhom_lop", "ma_gv", "cau_hoi",
        "noi_dung", "sentiment", "emotion", "topic", "target"]

df_out = df[COLS]
df_out.to_excel(PATHS["data_final"], index=False)
print(f"\n✓ Đã lưu: Du_Lieu_Chuan_Final.xlsx  ({len(df_out)} dòng)")

# Tạo gold set 1000 dòng đầu (để gán nhãn thủ công)
gold = df_out.head(1000).copy()
gold.to_excel(PATHS["gold_set"], index=False)
print(f"✓ Đã lưu: Gold_Set_1000.xlsx  (1000 dòng - dùng để gán nhãn thủ công)")

# ============================================================
# 5. GHI FILE QUY TẮC GÁN NHÃN
# ============================================================
rules = """
=============================================================================
QUY TẮC GÁN NHÃN THỦ CÔNG - HỆ THỐNG PHÂN TÍCH PHẢN HỒI SINH VIÊN
=============================================================================
Phiên bản: 1.0  |  Trường: Đại học Nha Trang

MỤC ĐÍCH: Đảm bảo tính nhất quán khi nhiều người cùng gán nhãn ("gold set").
Khi không chắc chắn, người gán nhãn nên chọn nhãn gần nhất và ghi chú.

──────────────────────────────────────────────────────────────────────────────
A. NHÃN SENTIMENT (Cảm xúc tổng thể)
──────────────────────────────────────────────────────────────────────────────
┌─────────────┬──────────────────────────────────────────────────────────────────┐
│ Nhãn        │ Định nghĩa                                                        │
├─────────────┼──────────────────────────────────────────────────────────────────┤
│ Positive    │ Phản hồi thể hiện sự hài lòng, khen ngợi, đánh giá tốt.          │
│             │ VD: "Cô dạy rất nhiệt tình, dễ hiểu"                             │
│             │ VD: "Thầy rất tận tâm và hỗ trợ SV ngoài giờ"                   │
├─────────────┼──────────────────────────────────────────────────────────────────┤
│ Neutral     │ Phản hồi trung lập: đề xuất cải thiện, không bày tỏ cảm xúc rõ  │
│             │ ràng, hoặc vừa khen vừa góp ý.                                   │
│             │ VD: "Mong thầy cho thêm bài tập thực hành"                       │
│             │ VD: "Cô dạy tốt nhưng tốc độ hơi nhanh"                          │
├─────────────┼──────────────────────────────────────────────────────────────────┤
│ Negative    │ Phản hồi thể hiện sự không hài lòng, phàn nàn, chỉ trích.        │
│             │ VD: "Thầy giải thích không rõ ràng, khó theo kịp"                │
│             │ VD: "Phòng học quá nóng, quạt hỏng mà không được sửa"            │
└─────────────┴──────────────────────────────────────────────────────────────────┘

LƯU Ý:
- Ưu tiên nội dung tổng thể, không chỉ 1-2 từ đơn lẻ.
- Câu có cả khen lẫn góp ý nhẹ → thường là Neutral.
- Câu chỉ có đề xuất (mong/nên/cần) mà không có chê → Neutral.

──────────────────────────────────────────────────────────────────────────────
B. NHÃN EMOTION (Loại cảm xúc chi tiết)
──────────────────────────────────────────────────────────────────────────────
┌──────────────┬───────────────────────────────────────────────────────────────────┐
│ Nhãn         │ Định nghĩa + Ví dụ minh họa                                       │
├──────────────┼───────────────────────────────────────────────────────────────────┤
│ KhenNgoi     │ Trực tiếp khen ngợi, ca ngợi giảng viên hoặc nhà trường.          │
│              │ VD: "Thầy giảng rất hay, xuất sắc"                                │
│              │ VD: "Cô rất dễ thương và nhiệt tình"                              │
├──────────────┼───────────────────────────────────────────────────────────────────┤
│ HaiLong      │ Hài lòng chung chung, không khen cụ thể nhưng thể hiện tích cực.  │
│              │ VD: "Học phần ổn, không có vấn đề gì"                             │
│              │ VD: "Nhìn chung em hài lòng với cách dạy của cô"                  │
├──────────────┼───────────────────────────────────────────────────────────────────┤
│ DeXuat       │ Đề xuất, góp ý mang tính xây dựng, không mang cảm xúc tiêu cực.  │
│              │ VD: "Mong thầy dạy chậm hơn một chút"                             │
│              │ VD: "Nên bổ sung thêm ví dụ thực tế vào bài giảng"               │
├──────────────┼───────────────────────────────────────────────────────────────────┤
│ PhanNan      │ Phàn nàn, không hài lòng, chỉ trích nhưng chưa đến mức thất vọng.│
│              │ VD: "Cô dạy hơi khó hiểu, lòng vòng"                             │
│              │ VD: "Phòng học quá nóng, mất tập trung"                           │
├──────────────┼───────────────────────────────────────────────────────────────────┤
│ ThatVong     │ Thất vọng nặng, thể hiện sự bức xúc, mất lòng tin.               │
│              │ VD: "Rất thất vọng với cách dạy của thầy"                        │
│              │ VD: "Không thể chấp nhận được, quá tệ"                           │
├──────────────┼───────────────────────────────────────────────────────────────────┤
│ KhongYKien   │ Không có ý kiến, phản hồi trống hoặc quá ngắn vô nghĩa.          │
│              │ VD: "Không", "OK", "...", ""                                       │
│              │ VD: "Bình thường thôi ạ", "Không có gì"                           │
└──────────────┴───────────────────────────────────────────────────────────────────┘

LƯU Ý:
- PhanNan dùng khi không hài lòng nhưng vẫn mang tính góp ý.
- ThatVong dùng khi có từ ngữ nặng nề, mất lòng tin hoàn toàn.
- Nếu câu vừa DeXuat vừa PhanNan → ưu tiên PhanNan.

──────────────────────────────────────────────────────────────────────────────
C. NHÃN TOPIC (Chủ đề phản hồi)
──────────────────────────────────────────────────────────────────────────────
┌─────────────────┬──────────────────────────────────────────────────────────────────┐
│ Nhãn            │ Định nghĩa + Ví dụ minh họa                                       │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ GiangDay        │ Liên quan đến phương pháp giảng dạy, nội dung môn học,             │
│                 │ kỹ năng giải thích của GV.                                         │
│                 │ VD: "Cô truyền đạt kiến thức rõ ràng, dễ hiểu"                   │
│                 │ VD: "Thầy giảng hơi nhanh, em khó theo kịp"                       │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ TaiLieu         │ Liên quan đến slide, giáo trình, tài liệu học tập, LMS.            │
│                 │ VD: "Mong thầy upload slide lên LMS sớm hơn"                       │
│                 │ VD: "Giáo trình khó tìm, mong cô chia sẻ link"                    │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ CoSoVatChat     │ Liên quan đến phòng học, thiết bị, wifi, điều hòa, âm thanh.      │
│                 │ VD: "Phòng học quá nóng, điều hòa hỏng"                           │
│                 │ VD: "Máy chiếu tối, khó nhìn bảng từ xa"                          │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ ThoiGian        │ Liên quan đến giờ giấc, đúng giờ, bù giờ, nghỉ học.               │
│                 │ VD: "Thầy hay đi trễ 10-15 phút"                                   │
│                 │ VD: "Mong cô thông báo trước nếu phải đổi lịch"                   │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ DanhGiaCongBang │ Liên quan đến điểm số, kiểm tra, đánh giá kết quả học tập.        │
│                 │ VD: "Được kiểm tra đánh giá công bằng, đúng thực chất"            │
│                 │ VD: "Tiêu chí chấm điểm chưa rõ ràng"                             │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ ThaiDo          │ Liên quan đến thái độ, cách ứng xử, giao tiếp của GV với SV.      │
│                 │ VD: "Thầy rất thân thiện, luôn lắng nghe SV"                      │
│                 │ VD: "Cô đôi khi có thái độ không tốt với SV"                      │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ Khac            │ Không thuộc các chủ đề trên, hoặc quá chung chung.                │
│                 │ VD: "Không có gì để góp ý"                                         │
│                 │ VD: "Tiếp tục phát huy" (không rõ chủ đề cụ thể)                  │
└─────────────────┴──────────────────────────────────────────────────────────────────┘

LƯU Ý:
- Chọn chủ đề CHỦ ĐẠO của câu phản hồi (không cần bao quát hết).
- Nếu câu đề cập cả GiangDay lẫn TaiLieu → chọn cái nổi bật hơn.

──────────────────────────────────────────────────────────────────────────────
D. QUY TẮC TỔNG QUÁT KHI GÁN NHÃN
──────────────────────────────────────────────────────────────────────────────
1. Đọc toàn bộ câu phản hồi trước, không đọc từng từ riêng lẻ.
2. Cân nhắc loại câu hỏi (ưu điểm / góp ý GV / góp ý nhà trường).
3. Không gán nhãn theo cảm tính — phải có lý do dựa trên nội dung.
4. Khi gặp phản hồi mơ hồ: gán Neutral / DeXuat / Khac.
5. Ghi chú vào cột "ghi_chu" khi phản hồi không rõ ràng.
6. QUAN TRỌNG: Đồng thuận ≥ 80% giữa 2 người gán nhãn mới tính hợp lệ.

──────────────────────────────────────────────────────────────────────────────
E. VÍ DỤ MINH HỌA TỔNG HỢP
──────────────────────────────────────────────────────────────────────────────
Phản hồi: "Cô dạy rất nhiệt tình, truyền đạt kiến thức rõ ràng, dễ hiểu"
→ sentiment: Positive | emotion: KhenNgoi | topic: GiangDay

Phản hồi: "Mong thầy upload tài liệu lên LMS trước buổi học"
→ sentiment: Neutral | emotion: DeXuat | topic: TaiLieu

Phản hồi: "Phòng học quá nóng, máy chiếu hay bị hỏng"
→ sentiment: Negative | emotion: PhanNan | topic: CoSoVatChat

Phản hồi: "không có gì"
→ sentiment: Neutral | emotion: KhongYKien | topic: Khac

Phản hồi: "Rất thất vọng với cách chấm điểm, không công bằng chút nào"
→ sentiment: Negative | emotion: ThatVong | topic: DanhGiaCongBang
=============================================================================
"""

with open(PATHS["label_rules"], "w", encoding="utf-8") as f:
    f.write(rules)
print("✓ Đã lưu: Quy_Tac_Gan_Nhan.txt")
print("\n[HOÀN TẤT BƯỚC 0]")
