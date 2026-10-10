"""
=============================================================================
BƯỚC 3 (V8): XÂY DỰNG ONTOLOGY MỞ RỘNG — Quản lý phản hồi sinh viên NTU
=============================================================================
Phiên bản V8 bổ sung đầy đủ:
  Classes mới:
    Department, Faculty, SeverityLevel*, Recommendation,
    FeedbackSource*, DataOrigin
  Object Properties mới:
    hasSeverityLevel, recommendsAction, belongsToAcademicTerm,
    belongsToDepartment, belongsToFaculty, generatedFromSource*
  Data Properties mới:
    hasAcademicYear*, hasPriorityScore, hasSourceName, hasResolutionStatus

  (* đã có skeleton trong V7 nhưng chưa có individuals/triples)

Output: ontology/Ontology_NTU_FeedbackSystem_V8.owl
=============================================================================
"""

import sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from config import PATHS

import pandas as pd
import re
import os

print("=" * 65)
print("BƯỚC 3: XÂY DỰNG ONTOLOGY V8 — MỞ RỘNG (OWL RDF/XML)")
print("=" * 65)

# ─── 1. ĐỌC DỮ LIỆU ─────────────────────────────────────────────────────────
try:
    df = pd.read_excel(PATHS["data_human"])
    col_map = {"Stt":"stt","NoiDung":"noi_dung","Sentiment":"sentiment",
               "Emotion":"emotion","Topic":"topic",
               "CauHoi":"cau_hoi","Lecturer":"target"}
    df.rename(columns=col_map, inplace=True)
except FileNotFoundError:
    df = pd.read_excel(PATHS["data_ntu_official"])
    assert df["nguon_goc"].isin(["NTU_GỐCC","NTU_TÁI_GÁN"]).all(), \
        "Phát hiện dữ liệu augmented – kiểm tra nguon_goc!"

df = df.dropna(subset=["noi_dung"]).copy()
df["stt"] = range(1, len(df)+1)
print(f"✓ Đọc {len(df)} phản hồi")

# ─── 2. HELPER ───────────────────────────────────────────────────────────────
def safe_id(s: str) -> str:
    return re.sub(r"[^\w]", "_", str(s).strip())[:80]

def xml_escape(s: str) -> str:
    return str(s).replace("&","&amp;").replace("<","&lt;")\
                 .replace(">","&gt;").replace('"',"&quot;")[:500]

NS   = "http://ntu.edu.vn/feedback_system#"
BASE = "http://ntu.edu.vn/feedback_system"

lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<rdf:RDF xmlns="http://ntu.edu.vn/feedback_system#"',
    '         xml:base="http://ntu.edu.vn/feedback_system"',
    '         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
    '         xmlns:owl="http://www.w3.org/2002/07/owl#"',
    '         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
    '         xmlns:xsd="http://www.w3.org/2001/XMLSchema#"',
    '         xmlns:ntu="http://ntu.edu.vn/feedback_system#">',
    '',
    f'    <owl:Ontology rdf:about="{BASE}">',
    '        <rdfs:label>NTU Feedback System Ontology V8 (Extended)</rdfs:label>',
    '        <rdfs:comment>Ontology mở rộng: quản lý phản hồi sinh viên - ĐH Nha Trang V8</rdfs:comment>',
    '    </owl:Ontology>', '',
]

# ─── 3. CLASSES ──────────────────────────────────────────────────────────────
# 3a. Classes lõi (kế thừa V7)
CLASSES_CORE = [
    ("Feedback",     "Phản hồi mở của sinh viên về học phần"),
    ("Course",       "Học phần được đánh giá"),
    ("Lecturer",     "Giảng viên được đề cập trong phản hồi"),
    ("QuestionType", "Loại câu hỏi: UuDiem / GopYGV / GopYNT"),
    ("Sentiment",    "Cảm xúc tổng thể: Positive/Neutral/Negative"),
    ("Emotion",      "Cảm xúc chi tiết: KhenNgoi/HaiLong/DeXuat/PhanNan/ThatVong/KhongYKien"),
    ("Topic",        "Chủ đề: GiangDay/TaiLieu/CoSoVatChat/ThoiGian/DanhGiaCongBang/ThaiDo/Khac"),
    ("Suggestion",   "Góp ý cải tiến trích xuất từ phản hồi"),
    ("AcademicTerm", "Kỳ học: HK1_2023_2024, HK2_2023_2024, ..."),
]

# 3b. Classes mở rộng V8
CLASSES_EXT = [
    ("Department",      "Bộ môn/Khoa chuyên ngành (ví dụ: Khoa CNTT, Khoa Kinh tế)"),
    ("Faculty",         "Đơn vị cấp trường / khoa mẹ quản lý nhiều Department"),
    ("SeverityLevel",   "Mức độ nghiêm trọng: Low / Medium / High / Critical"),
    ("Recommendation",  "Khuyến nghị hành động cụ thể phát sinh từ phân tích phản hồi"),
    ("FeedbackSource",  "Kênh/phương thức thu thập phản hồi: LMS, Phiếu giấy, Form online"),
    ("DataOrigin",      "Xuất xứ dữ liệu: phân biệt NTU gốc, tái gán nhãn hay augmented"),
]

lines.append("    <!-- ========== CLASSES (CORE – V7) ========== -->")
for name, comment in CLASSES_CORE:
    lines += [
        f'    <owl:Class rdf:about="{NS}{name}">',
        f'        <rdfs:label xml:lang="vi">{xml_escape(name)}</rdfs:label>',
        f'        <rdfs:comment xml:lang="vi">{xml_escape(comment)}</rdfs:comment>',
        '    </owl:Class>', '',
    ]

lines.append("    <!-- ========== CLASSES (EXTENDED – V8) ========== -->")
for name, comment in CLASSES_EXT:
    lines += [
        f'    <owl:Class rdf:about="{NS}{name}">',
        f'        <rdfs:label xml:lang="vi">{xml_escape(name)}</rdfs:label>',
        f'        <rdfs:comment xml:lang="vi">{xml_escape(comment)}</rdfs:comment>',
        '    </owl:Class>', '',
    ]

# ─── 4. OBJECT PROPERTIES ────────────────────────────────────────────────────
# format: (name, domain, range, comment)
OBJ_PROPS_CORE = [
    ("aboutCourse",          "Feedback",    "Course",       "Phản hồi thuộc học phần nào"),
    ("aboutLecturer",        "Feedback",    "Lecturer",     "Phản hồi đề cập giảng viên nào"),
    ("hasSentiment",         "Feedback",    "Sentiment",    "Cảm xúc tổng thể"),
    ("hasEmotion",           "Feedback",    "Emotion",      "Cảm xúc chi tiết"),
    ("aboutTopic",           "Feedback",    "Topic",        "Chủ đề được đề cập"),
    ("hasQuestionType",      "Feedback",    "QuestionType", "Loại câu hỏi khảo sát"),
    ("hasSuggestion",        "Feedback",    "Suggestion",   "Góp ý cải tiến"),
]

OBJ_PROPS_EXT = [
    ("hasSeverityLevel",     "Feedback",    "SeverityLevel",  "Mức độ nghiêm trọng của phản hồi (tự động hoặc chuyên gia gán)"),
    ("recommendsAction",     "Feedback",    "Recommendation", "Khuyến nghị hành động được đề xuất từ phản hồi này"),
    ("belongsToAcademicTerm","Feedback",    "AcademicTerm",   "Phản hồi thuộc kỳ học nào"),
    ("belongsToDepartment",  "Course",      "Department",     "Học phần thuộc bộ môn/khoa nào (Course → Department)"),
    ("belongsToFaculty",     "Department",  "Faculty",        "Bộ môn/khoa thuộc đơn vị cấp trường nào"),
    ("generatedFromSource",  "Feedback",    "FeedbackSource", "Phản hồi được thu thập qua kênh nào"),
    ("hasDataOrigin",        "Feedback",    "DataOrigin",     "Xuất xứ dữ liệu của phản hồi: NTU_GOC / TAI_GAN / AUG"),
    ("lecturerBelongsToDept","Lecturer",    "Department",     "Giảng viên thuộc bộ môn/khoa nào"),
]

lines.append("    <!-- ========== OBJECT PROPERTIES (CORE) ========== -->")
for name, domain, rng, comment in OBJ_PROPS_CORE:
    lines += [
        f'    <owl:ObjectProperty rdf:about="{NS}{name}">',
        f'        <rdfs:label xml:lang="vi">{xml_escape(name)}</rdfs:label>',
        f'        <rdfs:comment xml:lang="vi">{xml_escape(comment)}</rdfs:comment>',
        f'        <rdfs:domain rdf:resource="{NS}{domain}"/>',
        f'        <rdfs:range rdf:resource="{NS}{rng}"/>',
        '    </owl:ObjectProperty>', '',
    ]

lines.append("    <!-- ========== OBJECT PROPERTIES (EXTENDED – V8) ========== -->")
for name, domain, rng, comment in OBJ_PROPS_EXT:
    lines += [
        f'    <owl:ObjectProperty rdf:about="{NS}{name}">',
        f'        <rdfs:label xml:lang="vi">{xml_escape(name)}</rdfs:label>',
        f'        <rdfs:comment xml:lang="vi">{xml_escape(comment)}</rdfs:comment>',
        f'        <rdfs:domain rdf:resource="{NS}{domain}"/>',
        f'        <rdfs:range rdf:resource="{NS}{rng}"/>',
        '    </owl:ObjectProperty>', '',
    ]

# ─── 5. DATA PROPERTIES ──────────────────────────────────────────────────────
DATA_PROPS_CORE = [
    ("hasContent",      "Feedback",     "string",  "Nội dung văn bản phản hồi gốc"),
    ("hasCourseCode",   "Course",       "string",  "Mã học phần: INS330"),
    ("hasLecturerCode", "Lecturer",     "string",  "Mã giảng viên"),
    ("hasGroupCode",    "Feedback",     "string",  "Mã nhóm lớp học phần"),
    ("hasQuestionText", "QuestionType", "string",  "Nội dung câu hỏi khảo sát"),
    ("hasConfidence",   "Feedback",     "float",   "Độ tin cậy dự đoán NLP (0.0–1.0)"),
]

DATA_PROPS_EXT = [
    ("hasAcademicYear",    "AcademicTerm",   "string",  "Năm học dạng chuỗi, ví dụ: 2023-2024"),
    ("hasSemester",        "AcademicTerm",   "integer", "Số kỳ học: 1 hoặc 2"),
    ("hasPriorityScore",   "Recommendation", "float",   "Điểm ưu tiên xử lý (0.0 = thấp, 1.0 = khẩn cấp)"),
    ("hasSourceName",      "FeedbackSource", "string",  "Tên mô tả kênh thu thập phản hồi"),
    ("hasOriginLabel",     "DataOrigin",     "string",  "Nhãn xuất xứ: NTU_GOC / TAI_GAN / AUG_TEMPLATE / AUG_BACK"),
    ("hasResolutionStatus","Recommendation", "string",  "Trạng thái xử lý: Open / InProgress / Resolved / Closed"),
    ("hasActionText",      "Recommendation", "string",  "Nội dung hành động khuyến nghị cụ thể"),
    ("hasDepartmentCode",  "Department",     "string",  "Mã bộ môn/khoa"),
    ("hasFacultyCode",     "Faculty",        "string",  "Mã đơn vị cấp trường"),
]

lines.append("    <!-- ========== DATA PROPERTIES (CORE) ========== -->")
for name, domain, dtype, comment in DATA_PROPS_CORE:
    lines += [
        f'    <owl:DatatypeProperty rdf:about="{NS}{name}">',
        f'        <rdfs:label xml:lang="vi">{xml_escape(name)}</rdfs:label>',
        f'        <rdfs:comment xml:lang="vi">{xml_escape(comment)}</rdfs:comment>',
        f'        <rdfs:domain rdf:resource="{NS}{domain}"/>',
        f'        <rdfs:range rdf:resource="xsd:{dtype}"/>',
        '    </owl:DatatypeProperty>', '',
    ]

lines.append("    <!-- ========== DATA PROPERTIES (EXTENDED – V8) ========== -->")
for name, domain, dtype, comment in DATA_PROPS_EXT:
    lines += [
        f'    <owl:DatatypeProperty rdf:about="{NS}{name}">',
        f'        <rdfs:label xml:lang="vi">{xml_escape(name)}</rdfs:label>',
        f'        <rdfs:comment xml:lang="vi">{xml_escape(comment)}</rdfs:comment>',
        f'        <rdfs:domain rdf:resource="{NS}{domain}"/>',
        f'        <rdfs:range rdf:resource="xsd:{dtype}"/>',
        '    </owl:DatatypeProperty>', '',
    ]

# ─── 6. FIXED INDIVIDUALS ────────────────────────────────────────────────────
lines.append("    <!-- ========== INDIVIDUALS: Sentiment / Emotion / Topic / QuestionType ========== -->")

for s in ["Positive","Neutral","Negative"]:
    lines += [f'    <owl:NamedIndividual rdf:about="{NS}Sent_{s}">',
              f'        <rdf:type rdf:resource="{NS}Sentiment"/>',
              f'        <rdfs:label>{xml_escape(s)}</rdfs:label>',
              '    </owl:NamedIndividual>', '']

for e in ["KhenNgoi","HaiLong","DeXuat","PhanNan","ThatVong","KhongYKien"]:
    lines += [f'    <owl:NamedIndividual rdf:about="{NS}Emo_{e}">',
              f'        <rdf:type rdf:resource="{NS}Emotion"/>',
              f'        <rdfs:label>{xml_escape(e)}</rdfs:label>',
              '    </owl:NamedIndividual>', '']

for t in ["GiangDay","TaiLieu","CoSoVatChat","ThoiGian","DanhGiaCongBang","ThaiDo","Khac"]:
    lines += [f'    <owl:NamedIndividual rdf:about="{NS}Topic_{t}">',
              f'        <rdf:type rdf:resource="{NS}Topic"/>',
              f'        <rdfs:label>{xml_escape(t)}</rdfs:label>',
              '    </owl:NamedIndividual>', '']

for qid, qtxt in {
    "UuDiem": "Những ưu điểm nổi bật của GV trong quá trình giảng dạy học phần:",
    "GopYGV": "Những góp ý cho GV nhằm nâng cao hơn nữa chất lượng giảng dạy:",
    "GopYNT": "Những góp ý cho Nhà trường nhằm nâng cao chất lượng giảng dạy:",
}.items():
    lines += [f'    <owl:NamedIndividual rdf:about="{NS}QType_{qid}">',
              f'        <rdf:type rdf:resource="{NS}QuestionType"/>',
              f'        <ntu:hasQuestionText>{xml_escape(qtxt)}</ntu:hasQuestionText>',
              '    </owl:NamedIndividual>', '']

# ─── 7. INDIVIDUALS: SeverityLevel ───────────────────────────────────────────
lines.append("    <!-- ========== INDIVIDUALS: SeverityLevel ========== -->")
for sev_id, sev_label, sev_desc in [
    ("Low",      "Thấp",      "Phản hồi thông thường, chưa cần hành động ngay"),
    ("Medium",   "Trung bình","Cần theo dõi và xem xét trong kỳ tới"),
    ("High",     "Cao",       "Cần hành động sớm trong học kỳ hiện tại"),
    ("Critical", "Nghiêm trọng","Cần phản hồi ngay lập tức từ ban quản lý"),
]:
    lines += [
        f'    <owl:NamedIndividual rdf:about="{NS}Sev_{sev_id}">',
        f'        <rdf:type rdf:resource="{NS}SeverityLevel"/>',
        f'        <rdfs:label>{xml_escape(sev_label)}</rdfs:label>',
        f'        <rdfs:comment>{xml_escape(sev_desc)}</rdfs:comment>',
        '    </owl:NamedIndividual>', ''
    ]

# ─── 8. INDIVIDUALS: FeedbackSource ──────────────────────────────────────────
lines.append("    <!-- ========== INDIVIDUALS: FeedbackSource ========== -->")
for src_id, src_name, src_desc in [
    ("LMS_NTU",     "Hệ thống LMS NTU",   "Khảo sát điện tử qua cổng học tập LMS của trường"),
    ("PaperSurvey", "Phiếu khảo sát giấy","Phiếu giấy phát trực tiếp, số hóa thủ công"),
    ("OnlineForm",  "Form khảo sát online","Google Form / Microsoft Forms gửi qua email/zalo"),
]:
    lines += [
        f'    <owl:NamedIndividual rdf:about="{NS}Src_{src_id}">',
        f'        <rdf:type rdf:resource="{NS}FeedbackSource"/>',
        f'        <ntu:hasSourceName>{src_name}</ntu:hasSourceName>',
        f'        <rdfs:comment>{xml_escape(src_desc)}</rdfs:comment>',
        f'        <rdfs:label>{xml_escape(src_name)}</rdfs:label>',
        '    </owl:NamedIndividual>', ''
    ]

# ─── 9. INDIVIDUALS: DataOrigin ──────────────────────────────────────────────
lines.append("    <!-- ========== INDIVIDUALS: DataOrigin ========== -->")
for orig_id, orig_label, orig_desc in [
    ("NTU_GOC",       "NTU gốc",          "Dữ liệu thực tế do sinh viên NTU cung cấp, gán nhãn bởi con người"),
    ("NTU_TAI_GAN",   "NTU tái gán nhãn", "Dữ liệu NTU gốc được xem xét lại và gán lại nhãn theo quy tắc V7"),
    ("AUG_TEMPLATE",  "Augmented template","Câu tạo ra bằng template từ dữ liệu gốc để cân bằng lớp"),
    ("AUG_BACKTRANS", "Augmented backtranslation","Câu tạo ra bằng phương pháp back-translation"),
]:
    lines += [
        f'    <owl:NamedIndividual rdf:about="{NS}Orig_{orig_id}">',
        f'        <rdf:type rdf:resource="{NS}DataOrigin"/>',
        f'        <ntu:hasOriginLabel>{xml_escape(orig_label)}</ntu:hasOriginLabel>',
        f'        <rdfs:comment>{xml_escape(orig_desc)}</rdfs:comment>',
        f'        <rdfs:label>{xml_escape(orig_label)}</rdfs:label>',
        '    </owl:NamedIndividual>', ''
    ]

# ─── 10. INDIVIDUALS: AcademicTerm ───────────────────────────────────────────
lines.append("    <!-- ========== INDIVIDUALS: AcademicTerm ========== -->")
for term_id, year, sem, label in [
    ("HK1_2022_2023", "2022-2023", "1", "Học kỳ 1 năm 2022-2023"),
    ("HK2_2022_2023", "2022-2023", "2", "Học kỳ 2 năm 2022-2023"),
    ("HK1_2023_2024", "2023-2024", "1", "Học kỳ 1 năm 2023-2024"),
    ("HK2_2023_2024", "2023-2024", "2", "Học kỳ 2 năm 2023-2024"),
    ("HK1_2024_2025", "2024-2025", "1", "Học kỳ 1 năm 2024-2025"),
]:
    lines += [
        f'    <owl:NamedIndividual rdf:about="{NS}Term_{term_id}">',
        f'        <rdf:type rdf:resource="{NS}AcademicTerm"/>',
        f'        <ntu:hasAcademicYear>{year}</ntu:hasAcademicYear>',
        f'        <ntu:hasSemester>{sem}</ntu:hasSemester>',
        f'        <rdfs:label>{xml_escape(label)}</rdfs:label>',
        '    </owl:NamedIndividual>', ''
    ]

# ─── 11. INDIVIDUALS: Faculty ────────────────────────────────────────────────
lines.append("    <!-- ========== INDIVIDUALS: Faculty ========== -->")
FACULTIES = {
    "KhoaCNTT":     ("CNTT",   "Khoa Công nghệ Thông tin"),
    "KhoaKinhTe":   ("KKTE",   "Khoa Kinh tế"),
    "KhoaCoBan":    ("KKCB",   "Khoa Khoa học Cơ bản"),
    "KhoaNgoaiNgu": ("KNN",    "Khoa Ngoại ngữ"),
    "KhoaNuoiTrong":("KNTS",   "Khoa Nuôi trồng Thủy sản"),
    "KhoaCongNghe": ("KCN",    "Khoa Công nghệ Thực phẩm"),
    "KhoaKinhTeNN": ("KKTNN",  "Khoa Kinh tế & Nông nghiệp"),
}
for fac_id, (fac_code, fac_name) in FACULTIES.items():
    lines += [
        f'    <owl:NamedIndividual rdf:about="{NS}Fac_{fac_id}">',
        f'        <rdf:type rdf:resource="{NS}Faculty"/>',
        f'        <ntu:hasFacultyCode>{fac_code}</ntu:hasFacultyCode>',
        f'        <rdfs:label>{xml_escape(fac_name)}</rdfs:label>',
        '    </owl:NamedIndividual>', ''
    ]

# ─── 12. INDIVIDUALS: Department ─────────────────────────────────────────────
lines.append("    <!-- ========== INDIVIDUALS: Department ========== -->")
# Each department links to its Faculty
DEPARTMENTS = [
    # (dept_id, dept_code, dept_name, parent_faculty_id)
    ("BM_KTPM",   "KTPM", "Bộ môn Kỹ thuật Phần mềm",          "KhoaCNTT"),
    ("BM_HTTT",   "HTTT", "Bộ môn Hệ thống Thông tin",          "KhoaCNTT"),
    ("BM_MMTT",   "MMTT", "Bộ môn Mạng máy tính & Truyền thông","KhoaCNTT"),
    ("BM_KTTT",   "KTTT", "Bộ môn Khoa học Máy tính",           "KhoaCNTT"),
    ("BM_KT",     "KT",   "Bộ môn Kế toán",                     "KhoaKinhTe"),
    ("BM_QTKD",   "QTKD", "Bộ môn Quản trị Kinh doanh",         "KhoaKinhTe"),
    ("BM_Toan",   "TOAN", "Bộ môn Toán",                        "KhoaCoBan"),
    ("BM_LyHoa",  "LYHO", "Bộ môn Lý-Hóa",                     "KhoaCoBan"),
    ("BM_TA",     "TA",   "Bộ môn Tiếng Anh",                   "KhoaNgoaiNgu"),
]
for dept_id, dept_code, dept_name, fac_id in DEPARTMENTS:
    lines += [
        f'    <owl:NamedIndividual rdf:about="{NS}Dept_{dept_id}">',
        f'        <rdf:type rdf:resource="{NS}Department"/>',
        f'        <ntu:hasDepartmentCode>{dept_code}</ntu:hasDepartmentCode>',
        f'        <ntu:belongsToFaculty rdf:resource="{NS}Fac_{fac_id}"/>',
        f'        <rdfs:label>{xml_escape(dept_name)}</rdfs:label>',
        '    </owl:NamedIndividual>', ''
    ]

# ─── 13. INDIVIDUALS: Recommendation ─────────────────────────────────────────
lines.append("    <!-- ========== INDIVIDUALS: Recommendation (ví dụ) ========== -->")
RECOMMENDATIONS = [
    ("Rec_GD_001", "0.85", "Open",       "GiangDay",    "Đa dạng hóa phương pháp giảng dạy, tăng bài tập thực hành"),
    ("Rec_TL_001", "0.72", "InProgress", "TaiLieu",     "Cập nhật slide bài giảng, bổ sung tài liệu tiếng Anh"),
    ("Rec_CS_001", "0.91", "Open",       "CoSoVatChat", "Nâng cấp phòng máy tính Lab C2, thay máy chiếu hỏng"),
    ("Rec_TG_001", "0.60", "Resolved",   "ThoiGian",    "Điều chỉnh lịch thi phù hợp, tránh dồn nhiều môn"),
    ("Rec_DG_001", "0.78", "InProgress", "DanhGiaCongBang","Chuẩn hóa tiêu chí chấm điểm, công bố trọng số rõ ràng"),
    ("Rec_TD_001", "0.55", "Open",       "ThaiDo",      "Tăng cường tương tác GV-SV trong giờ học"),
]
for rec_id, priority, status, topic, action in RECOMMENDATIONS:
    lines += [
        f'    <owl:NamedIndividual rdf:about="{NS}{rec_id}">',
        f'        <rdf:type rdf:resource="{NS}Recommendation"/>',
        f'        <ntu:hasPriorityScore rdf:datatype="xsd:float">{priority}</ntu:hasPriorityScore>',
        f'        <ntu:hasResolutionStatus>{status}</ntu:hasResolutionStatus>',
        f'        <ntu:hasActionText>{xml_escape(action)}</ntu:hasActionText>',
        f'        <rdfs:label>{xml_escape(rec_id)}</rdfs:label>',
        '    </owl:NamedIndividual>', ''
    ]

# ─── 14. INDIVIDUALS: Courses & Lecturers ─────────────────────────────────────
lines.append("    <!-- ========== COURSES & LECTURERS ========== -->")

# Course–Department mapping (mã học phần → bộ môn)
COURSE_DEPT_MAP = {
    "INS": "Dept_BM_KTPM", "SOT": "Dept_BM_HTTT", "NEC": "Dept_BM_MMTT",
    "MAT": "Dept_BM_Toan",  "COS": "Dept_BM_KTTT", "AQU": "Dept_BM_TA",
    "INT": "Dept_BM_KTPM",
}

courses = {}
if "hoc_phan" in df.columns:
    for hp in df["hoc_phan"].dropna().unique():
        cid = safe_id(hp)
        courses[hp] = cid
        prefix = str(hp)[:3].upper()
        dept_iri = COURSE_DEPT_MAP.get(prefix, "")
        dept_triple = f'        <ntu:belongsToDepartment rdf:resource="{NS}{dept_iri}"/>' if dept_iri else ""
        fb_lines = [
            f'    <owl:NamedIndividual rdf:about="{NS}Course_{cid}">',
            f'        <rdf:type rdf:resource="{NS}Course"/>',
            f'        <ntu:hasCourseCode>{xml_escape(hp)}</ntu:hasCourseCode>',
            f'        <rdfs:label>{xml_escape(hp)}</rdfs:label>',
        ]
        if dept_triple:
            fb_lines.append(dept_triple)
        fb_lines += ['    </owl:NamedIndividual>', '']
        lines.extend(fb_lines)

lecturers = {}
tgt_col = "target" if "target" in df.columns else "Lecturer"
for raw in df[tgt_col].dropna().unique():
    lid = safe_id(raw)
    lecturers[raw] = lid
    lines += [
        f'    <owl:NamedIndividual rdf:about="{NS}Lecturer_{lid}">',
        f'        <rdf:type rdf:resource="{NS}Lecturer"/>',
        f'        <ntu:hasLecturerCode>{xml_escape(raw)}</ntu:hasLecturerCode>',
        f'        <ntu:lecturerBelongsToDept rdf:resource="{NS}Dept_BM_KTPM"/>',
        f'        <rdfs:label>{xml_escape(raw)}</rdfs:label>',
        '    </owl:NamedIndividual>', ''
    ]

# ─── 15. INDIVIDUALS: Feedback ───────────────────────────────────────────────
lines.append("    <!-- ========== FEEDBACK INDIVIDUALS ========== -->")
print("→ Đang tạo Feedback individuals...")

q_map = {
    "Những ưu điểm nổi bật của GV trong quá trình giảng dạy học phần:": "UuDiem",
    "Những góp ý cho GV nhằm nâng cao hơn nữa chất lượng giảng dạy của học phần:": "GopYGV",
    "Những góp ý cho Nhà trường nhằm nâng cao hơn nữa chất lượng giảng dạy của học phần:": "GopYNT",
}

# Severity auto-assign based on emotion
SEV_BY_EMOTION = {
    "ThatVong": "Critical", "PhanNan": "High",
    "DeXuat": "Medium", "KhongYKien": "Low",
    "HaiLong": "Low", "KhenNgoi": "Low",
}

# Recommendation auto-link by topic
REC_BY_TOPIC = {
    "GiangDay": "Rec_GD_001", "TaiLieu": "Rec_TL_001",
    "CoSoVatChat": "Rec_CS_001", "ThoiGian": "Rec_TG_001",
    "DanhGiaCongBang": "Rec_DG_001", "ThaiDo": "Rec_TD_001",
}

for _, row in df.iterrows():
    fid     = f"FB_{int(row['stt'])}"
    sent    = str(row.get("sentiment","")).strip()
    emo     = str(row.get("emotion","")).strip()
    topic   = str(row.get("topic","")).strip()
    content = xml_escape(str(row["noi_dung"]))
    cau_hoi = str(row.get("cau_hoi","")).strip()
    q_type  = q_map.get(cau_hoi, "GopYGV")
    sev     = SEV_BY_EMOTION.get(emo, "Medium")

    # AcademicTerm auto-assign (use HK1_2023_2024 as default)
    term    = "Term_HK1_2023_2024"

    fb = [
        f'    <owl:NamedIndividual rdf:about="{NS}{fid}">',
        f'        <rdf:type rdf:resource="{NS}Feedback"/>',
        f'        <ntu:hasContent>{content}</ntu:hasContent>',
        f'        <ntu:hasSentiment rdf:resource="{NS}Sent_{sent}"/>',
        f'        <ntu:hasSeverityLevel rdf:resource="{NS}Sev_{sev}"/>',
        f'        <ntu:belongsToAcademicTerm rdf:resource="{NS}{term}"/>',
        f'        <ntu:generatedFromSource rdf:resource="{NS}Src_LMS_NTU"/>',
        f'        <ntu:hasDataOrigin rdf:resource="{NS}Orig_NTU_GOC"/>',
    ]
    if emo:
        fb.append(f'        <ntu:hasEmotion rdf:resource="{NS}Emo_{emo}"/>')
    if topic:
        fb.append(f'        <ntu:aboutTopic rdf:resource="{NS}Topic_{topic}"/>')
        rec = REC_BY_TOPIC.get(topic)
        if rec and sent == "Negative":
            fb.append(f'        <ntu:recommendsAction rdf:resource="{NS}{rec}"/>')
    fb.append(f'        <ntu:hasQuestionType rdf:resource="{NS}QType_{q_type}"/>')
    tgt = str(row.get(tgt_col,"")).strip()
    if tgt in lecturers:
        fb.append(f'        <ntu:aboutLecturer rdf:resource="{NS}Lecturer_{lecturers[tgt]}"/>')
    if "hoc_phan" in df.columns:
        hp = str(row.get("hoc_phan","")).strip()
        if hp in courses:
            fb.append(f'        <ntu:aboutCourse rdf:resource="{NS}Course_{courses[hp]}"/>')
    fb += ['    </owl:NamedIndividual>', '']
    lines.extend(fb)

lines.append("</rdf:RDF>")

# ─── 16. GHI FILE ─────────────────────────────────────────────────────────────
out_dir  = os.path.join(_ROOT, "ontology")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "Ontology_NTU_FeedbackSystem_V8.owl")
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

size_kb = os.path.getsize(out_file) // 1024
print(f"✓ Đã lưu: {out_file}  ({size_kb} KB)")
print(f"  - Classes core     : {len(CLASSES_CORE)}")
print(f"  - Classes extended : {len(CLASSES_EXT)}")
print(f"  - Obj Props core   : {len(OBJ_PROPS_CORE)}")
print(f"  - Obj Props ext    : {len(OBJ_PROPS_EXT)}")
print(f"  - Data Props core  : {len(DATA_PROPS_CORE)}")
print(f"  - Data Props ext   : {len(DATA_PROPS_EXT)}")
print(f"  - Feedbacks        : {len(df)}")
print(f"  - Courses          : {len(courses)}")
print(f"  - Lecturers        : {len(lecturers)}")
print(f"  - Faculties        : {len(FACULTIES)}")
print(f"  - Departments      : {len(DEPARTMENTS)}")
print(f"  - Recommendations  : {len(RECOMMENDATIONS)}")
print("\n[HOÀN TẤT BƯỚC 3 – V8 MỞ RỘNG]")
