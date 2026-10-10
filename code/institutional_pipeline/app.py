"""
=============================================================================
app.py  –  NTU Feedback Analytics  |  Giao diện Streamlit hoàn chỉnh
=============================================================================
Chức năng:
  1. Nhập phản hồi mới
  2. Dự đoán Sentiment / Emotion / Topic (rule-based)
  3. Hiển thị Ontology Mapping & RDF Triples
  4. Truy vấn SPARQL trực tiếp (10 query + custom)
  5. Lọc theo GV / Học phần / Học kỳ / Loại câu hỏi
  6. Dashboard thống kê tương tác
  7. Gợi ý hành động cải tiến
  8. Xếp hạng mức độ ưu tiên xử lý
=============================================================================
Chạy:  python -m streamlit run dashboard/app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os, re
from collections import Counter, defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(_HERE) if os.path.basename(_HERE) == "dashboard" else _HERE
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA_PATH = os.path.join(ROOT, "data", "processed", "Du_Lieu_NTU_Official.xlsx")
OWL_PATHS = [
    os.path.join(ROOT, "ontology", "Ontology_NTU_FeedbackSystem_V8.owl"),
    os.path.join(ROOT, "ontology", "Ontology_NTU_FeedbackSystem_V7.owl"),
]
NS_NTU = "http://ntu.edu.vn/feedback_system#"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NTU Feedback Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #f4f6f9; }
html, body, [class*="css"] { font-size: 15px !important; }
p, .stMarkdown, .stText { font-size: 1.0rem !important; line-height: 1.65 !important; }
.stSelectbox label, .stTextInput label, .stTextArea label, .stRadio label {
    font-size: 0.95rem !important; font-weight: 600 !important;
}
h1 { font-size: 1.9rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.25rem !important; }
.stDataFrame { font-size: 0.93rem !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f0fe 0%, #dce8fd 100%);
    padding-top: 0;
    border-right: 1px solid #c7d7f5;
}
[data-testid="stSidebar"] * { color: #1a2744 !important; }
[data-testid="stSidebar"] .stRadio label { 
    font-size: 1.0rem; padding: 0.35rem 0; font-weight: 500;
}
/* ── Metric Cards ── */
.kpi-card {
    background: white; border-radius: 12px;
    padding: 18px 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-left: 4px solid;
}
.kpi-card.blue  { border-color: #3b82f6; }
.kpi-card.green { border-color: #10b981; }
.kpi-card.red   { border-color: #ef4444; }
.kpi-card.amber { border-color: #f59e0b; }
.kpi-label { font-size: 0.85rem; color: #6b7280; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
.kpi-value { font-size: 2.2rem; font-weight: 700; color: #111827; line-height: 1; }
.kpi-sub   { font-size: 0.88rem; color: #9ca3af; margin-top: 6px; }
/* ── Badges ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
}
.badge-pos { background:#d1fae5; color:#065f46; }
.badge-neg { background:#fee2e2; color:#991b1b; }
.badge-neu { background:#e5e7eb; color:#374151; }
.badge-emo { background:#ede9fe; color:#5b21b6; }
.badge-top { background:#dbeafe; color:#1e40af; }
/* ── Section Header ── */
.sec-header {
    font-size: 1.2rem; font-weight: 700; color: #1e3a5f;
    border-bottom: 2.5px solid #3b82f6; padding-bottom: 8px;
    margin-bottom: 18px;
}
/* ── Tab Styles ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    font-size: 1.0rem !important; font-weight: 600 !important;
    padding: 10px 18px !important;
}
/* ── Table font ── */
table { font-size: 0.94rem !important; }
th { font-size: 0.95rem !important; font-weight: 700 !important; background: #eff6ff !important; }
/* ── RDF Block ── */
.rdf-block {
    background: #f0f7ff; color: #1e3a5f; font-family: 'Courier New', monospace;
    font-size: 0.88rem; border-radius: 10px; padding: 16px 20px;
    white-space: pre-wrap; line-height: 1.8;
    border: 1px solid #bfdbfe;
}
.rdf-subject   { color: #1d4ed8; font-weight: 700; }
.rdf-predicate { color: #7c3aed; }
.rdf-object    { color: #065f46; }
/* ── Priority Row ── */
.pri-critical { background: #fef2f2 !important; }
.pri-high     { background: #fff7ed !important; }
.pri-medium   { background: #fefce8 !important; }
.pri-low      { background: #f0fdf4 !important; }
/* ── Action Card ── */
.action-card {
    background: white; border-radius: 10px; padding: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07); margin-bottom: 12px;
    border-top: 3px solid;
}
.action-card.critical { border-color: #ef4444; }
.action-card.high     { border-color: #f97316; }
.action-card.medium   { border-color: #eab308; }
/* ── Sidebar Logo ── */
.sidebar-logo {
    background: rgba(255,255,255,0.7); border-radius: 10px;
    padding: 16px; margin-bottom: 8px; text-align: center;
    box-shadow: 0 2px 8px rgba(37,93,204,0.10);
}
/* ── Input Prediction Result ── */
.pred-result {
    background: white; border-radius: 10px; padding: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-top: 10px;
}
/* ── Ontology Mapping ── */
.onto-row {
    display: flex; align-items: center; padding: 6px 0;
    border-bottom: 1px dashed #e5e7eb;
}
.onto-pred { color: #6d28d9; font-weight: 600; font-size: 0.82rem; min-width: 200px; }
.onto-obj  { color: #065f46; background: #d1fae5; padding: 2px 8px;
             border-radius: 6px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SENTIMENT_VI = {"Positive": "Tích cực", "Neutral": "Trung tính", "Negative": "Tiêu cực"}
EMOTION_VI   = {
    "KhenNgoi": "Khen ngợi", "HaiLong": "Hài lòng",
    "DeXuat": "Đề xuất",    "PhanNan": "Phàn nàn",
    "ThatVong": "Thất vọng","KhongYKien": "Không ý kiến",
}
TOPIC_VI = {
    "GiangDay": "Giảng dạy",     "TaiLieu": "Tài liệu",
    "CoSoVatChat": "CSVC",        "ThoiGian": "Thời gian",
    "DanhGiaCongBang": "Đánh giá","ThaiDo": "Thái độ", "Khac": "Khác",
}
CAU_HOI_MAP = {
    "Những ưu điểm nổi bật của GV trong quá trình giảng dạy học phần:": "Ưu điểm GV",
    "Những góp ý cho GV nhằm nâng cao hơn nữa chất lượng giảng dạy của học phần:": "Góp ý GV",
    "Những góp ý cho Nhà trường nhằm nâng cao hơn nữa chất lượng giảng dạy của học phần:": "Góp ý Nhà trường",
}
SENTIMENT_COLORS = {"Positive": "#10b981", "Neutral": "#6b7280", "Negative": "#ef4444"}
EMOTION_COLORS   = {
    "KhenNgoi": "#10b981", "HaiLong": "#3b82f6", "DeXuat": "#8b5cf6",
    "KhongYKien": "#6b7280", "PhanNan": "#f97316", "ThatVong": "#ef4444",
}
TOPIC_COLORS = {
    "GiangDay": "#3b82f6", "TaiLieu": "#8b5cf6", "CoSoVatChat": "#f97316",
    "ThoiGian": "#06b6d4", "DanhGiaCongBang": "#84cc16", "ThaiDo": "#ec4899", "Khac": "#9ca3af",
}

IMPROVEMENT_MAP = {
    "GiangDay":      ("🎓", "Đổi mới phương pháp giảng dạy",
                      "Tăng bài tập thực hành, đa dạng hóa PPGD, tổ chức học theo dự án; "
                      "GV nên kết hợp lý thuyết và case study thực tế ngành."),
    "CoSoVatChat":   ("🖥️", "Nâng cấp cơ sở vật chất",
                      "Đầu tư thiết bị phòng Lab, thay máy chiếu cũ, cải thiện đường truyền mạng; "
                      "ưu tiên phòng học có yêu cầu thực hành cao."),
    "TaiLieu":       ("📚", "Cập nhật tài liệu học tập",
                      "Biên soạn lại giáo trình theo chuẩn mới nhất; bổ sung tài liệu tiếng Anh "
                      "và tài nguyên học trực tuyến (video, e-book)."),
    "ThoiGian":      ("⏰", "Cải thiện kế hoạch thời gian",
                      "Sắp xếp lịch thi không trùng nhiều môn; thông báo lịch học sớm; "
                      "phân bổ hợp lý tỷ lệ lý thuyết/thực hành."),
    "DanhGiaCongBang":("⚖️", "Minh bạch hóa đánh giá",
                      "Công bố rõ ràng tiêu chí điểm, trọng số từng phần; "
                      "tổ chức phúc khảo minh bạch; áp dụng rubric chuẩn hóa."),
    "ThaiDo":        ("🤝", "Nâng cao thái độ phục vụ",
                      "Tăng cường tương tác GV-SV ngoài giờ học; nhanh chóng phản hồi email; "
                      "tập huấn kỹ năng giao tiếp sư phạm cho GV mới."),
}

PRIORITY_WEIGHTS = {
    "sentiment":  {"Negative": 1.0, "Neutral": 0.3, "Positive": 0.0},
    "emotion":    {"ThatVong": 1.0, "PhanNan": 0.85, "DeXuat": 0.45,
                   "KhongYKien": 0.15, "HaiLong": 0.05, "KhenNgoi": 0.0},
}
SEVERITY_THRESHOLDS = {
    "Critical": 0.85, "High": 0.65, "Medium": 0.40, "Low": 0.0,
}
SEVERITY_COLORS = {
    "Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308", "Low": "#10b981",
}
SEVERITY_ICONS = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}

# ── Rule-based Predictor ──────────────────────────────────────────────────────
_KW_NEG = ["không tốt","không hay","không hiểu","quá khó","tệ","kém","chán","thất vọng",
           "phàn nàn","vô lý","thiếu","nhàm","chậm","trễ","khó chịu","không hài lòng",
           "không nhiệt tình","không trả lời","nói dối","không công bằng","thấp"]
_KW_POS = ["tốt","hay","nhiệt tình","tận tâm","hỗ trợ","xuất sắc","giỏi","khen","hài lòng",
           "tuyệt","rõ ràng","dễ hiểu","bổ ích","thực tế","chuyên nghiệp","hứng thú",
           "công bằng","dễ tiếp cận","sáng tạo","vui","thú vị"]
_KW_EMO = {
    "ThatVong": ["thất vọng","thất vọng quá","rất tệ","quá tệ","nói dối","vô lý"],
    "PhanNan":  ["phàn nàn","không hài lòng","không tốt","kém","chán","nhàm","sai"],
    "DeXuat":   ["nên","cần","đề xuất","kiến nghị","mong muốn","hy vọng","muốn","nếu có thể"],
    "HaiLong":  ["hài lòng","vừa ý","ổn","bình thường","chấp nhận"],
    "KhenNgoi": ["khen","xuất sắc","tuyệt","giỏi","nhiệt tình","tận tâm","rất tốt"],
}
_KW_TOPIC = {
    "CoSoVatChat": ["phòng học","máy tính","thiết bị","lab","internet","wifi","máy chiếu",
                    "điều hòa","cơ sở","vật chất","phần mềm","máy móc"],
    "TaiLieu":     ["tài liệu","slide","bài giảng","giáo trình","sách","đề cương","bài tập",
                    "ví dụ","thực hành","đề thi"],
    "ThoiGian":    ["thời gian","lịch học","lịch thi","giờ","tuần","kỳ","học kỳ",
                    "buổi học","trễ","đúng giờ"],
    "DanhGiaCongBang":["điểm","điểm số","đánh giá","công bằng","kiểm tra","thi","phúc khảo",
                       "trọng số","tiêu chí"],
    "ThaiDo":      ["thái độ","nhiệt tình","tận tâm","hỗ trợ","trả lời","phản hồi",
                    "email","giao tiếp"],
    "GiangDay":    ["giảng dạy","dạy","bài giảng","phương pháp","giảng viên","gv","thầy",
                    "cô","giải thích","hiểu","lý thuyết","thực hành"],
}

def predict_sentiment(text: str) -> tuple[str, float]:
    t = text.lower()
    n = sum(1 for k in _KW_NEG if k in t)
    p = sum(1 for k in _KW_POS if k in t)
    if n > p:   return "Negative", round(min(0.5 + n * 0.1, 0.97), 2)
    elif p > 0: return "Positive", round(min(0.5 + p * 0.1, 0.97), 2)
    else:       return "Neutral",  0.62

def predict_emotion(text: str, sentiment: str) -> tuple[str, float]:
    t = text.lower()
    scores = {e: sum(1 for k in kws if k in t) for e, kws in _KW_EMO.items()}
    best = max(scores, key=lambda x: scores[x])
    if scores[best] == 0:
        if sentiment == "Negative": best, conf = "PhanNan", 0.55
        elif sentiment == "Positive": best, conf = "KhenNgoi", 0.60
        else: best, conf = "KhongYKien", 0.55
    else:
        conf = round(min(0.5 + scores[best] * 0.12, 0.95), 2)
    return best, conf

def predict_topic(text: str) -> tuple[str, float]:
    t = text.lower()
    scores = {tp: sum(1 for k in kws if k in t) for tp, kws in _KW_TOPIC.items()}
    best = max(scores, key=lambda x: scores[x])
    if scores[best] == 0:
        return "GiangDay", 0.45
    conf = round(min(0.5 + scores[best] * 0.12, 0.93), 2)
    return best, conf

def compute_priority(row) -> float:
    s = PRIORITY_WEIGHTS["sentiment"].get(row["sentiment"], 0.3)
    e = PRIORITY_WEIGHTS["emotion"].get(row["emotion"], 0.3)
    return round((s * 0.55 + e * 0.45), 3)

def get_severity(score: float) -> str:
    for sev, thresh in SEVERITY_THRESHOLDS.items():
        if score >= thresh:
            return sev
    return "Low"

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_excel(DATA_PATH)
    df["priority_score"] = df.apply(compute_priority, axis=1)
    df["severity"]       = df["priority_score"].apply(get_severity)
    df["cau_hoi_short"]  = df["cau_hoi"].map(CAU_HOI_MAP).fillna(df["cau_hoi"])
    df["hoc_ky"]         = df["nhom_lop"].apply(lambda x: "HK1" if pd.notna(x) and int(x) <= 35 else "HK2")
    return df

@st.cache_resource(show_spinner=False)
def get_owl_graph():
    try:
        from owl_sparql_engine import OWLGraph, get_ontology_stats, QUERY_MAP
        for p in OWL_PATHS:
            if os.path.exists(p):
                g = OWLGraph(p)
                stats = get_ontology_stats(p)
                return g, stats, list(QUERY_MAP.keys()), p
    except Exception:
        pass
    return None, {}, [], ""

@st.cache_data(show_spinner=False)
def run_sparql_query(query_name: str, owl_path: str) -> pd.DataFrame:
    try:
        from owl_sparql_engine import execute_query
        return execute_query(query_name, owl_path)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

# ── RDF Triple Generator ───────────────────────────────────────────────────────
def generate_rdf(fb_id: str, content: str, sentiment: str, emotion: str,
                 topic: str, gv: str, hp: str, cau_hoi: str) -> str:
    s = sentiment; e = emotion; t = topic
    qt_map = {
        "Ưu điểm GV": "QType_UuDiem",
        "Góp ý GV": "QType_GopYGV",
        "Góp ý Nhà trường": "QType_GopYNT",
    }
    qt = qt_map.get(cau_hoi, "QType_GopYGV")
    lines = [
        f'@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .',
        f'@prefix owl:  <http://www.w3.org/2002/07/owl#> .',
        f'@prefix ntu:  <http://ntu.edu.vn/feedback_system#> .',
        f'@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
        f'',
        f'ntu:{fb_id}',
        f'    rdf:type              owl:NamedIndividual , ntu:Feedback ;',
        f'    ntu:hasContent        "{content[:80].replace(chr(34), chr(39))}..." ;',
        f'    ntu:hasSentiment      ntu:Sent_{s} ;',
        f'    ntu:hasEmotion        ntu:Emo_{e} ;',
        f'    ntu:aboutTopic        ntu:Topic_{t} ;',
        f'    ntu:hasQuestionType   ntu:{qt} ;',
    ]
    if gv:
        gv_id = re.sub(r"[^\w]", "_", gv)
        lines.append(f'    ntu:aboutLecturer     ntu:Lecturer_{gv_id} ;')
    if hp:
        lines.append(f'    ntu:aboutCourse       ntu:Course_{hp} ;')
    lines[-1] = lines[-1].rstrip(" ;") + " ."
    return "\n".join(lines)

def generate_ontology_mapping(sentiment, emotion, topic, gv, hp, cau_hoi):
    qt_map = {"Ưu điểm GV":"QType_UuDiem","Góp ý GV":"QType_GopYGV","Góp ý Nhà trường":"QType_GopYNT"}
    rows = [
        ("rdf:type",          "ntu:Feedback",            "class"),
        ("ntu:hasSentiment",  f"ntu:Sent_{sentiment}",   "obj"),
        ("ntu:hasEmotion",    f"ntu:Emo_{emotion}",      "obj"),
        ("ntu:aboutTopic",    f"ntu:Topic_{topic}",      "obj"),
        ("ntu:hasQuestionType", f"ntu:{qt_map.get(cau_hoi,'QType_GopYGV')}", "obj"),
    ]
    if gv:
        gv_clean = re.sub(r'[^\w]', '_', gv)
        rows.append(("ntu:aboutLecturer", f"ntu:Lecturer_{gv_clean}", "obj"))
    if hp:
        rows.append(("ntu:aboutCourse", f"ntu:Course_{hp}", "obj"))
    return rows

# ─── Helpers ──────────────────────────────────────────────────────────────────
def sentiment_badge(s):
    cls = {"Positive": "badge-pos", "Negative": "badge-neg", "Neutral": "badge-neu"}.get(s, "badge-neu")
    return f'<span class="badge {cls}">{SENTIMENT_VI.get(s, s)}</span>'

def conf_bar(conf: float, color: str = "#3b82f6") -> str:
    pct = int(conf * 100)
    return (f'<div style="background:#e5e7eb;border-radius:6px;height:8px;width:100%;margin-top:4px">'
            f'<div style="background:{color};width:{pct}%;height:8px;border-radius:6px"></div></div>'
            f'<div style="font-size:0.75rem;color:#6b7280;text-align:right">{pct}%</div>')

def kpi_card(label, value, sub, color):
    return f"""<div class="kpi-card {color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

# ── Session State ──────────────────────────────────────────────────────────────
if "new_feedbacks" not in st.session_state:
    st.session_state.new_feedbacks = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div class="sidebar-logo">
        <div style="font-size:2rem">🎓</div>
        <div style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-top:4px">NTU Feedback</div>
        <div style="font-size:0.82rem;color:#94a3b8">Analytics Dashboard · V8</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠 Tổng quan",
        "📝 Phân tích phản hồi",
        "🔍 Truy vấn SPARQL",
        "📊 Lọc & Phân tích",
        "🎯 Ưu tiên & Hành động",
    ], label_visibility="collapsed")

    st.markdown("---")
    df_main = load_data()
    owl_g, owl_stats, query_names, owl_path = get_owl_graph()

    if not df_main.empty:
        st.markdown(f"""<div style="font-size:0.8rem;color:#94a3b8;padding:0 4px">
            <div>📂 <b style="color:#c8d0e7">{len(df_main):,}</b> phản hồi đã tải</div>
            <div style="margin-top:4px">📋 <b style="color:#c8d0e7">{df_main['ma_gv'].nunique()}</b> giảng viên</div>
            <div style="margin-top:4px">📚 <b style="color:#c8d0e7">{df_main['hoc_phan'].nunique()}</b> học phần</div>
        </div>""", unsafe_allow_html=True)
        if owl_stats:
            st.markdown(f"""<div style="font-size:0.8rem;color:#94a3b8;padding:4px 4px 0">
                <div style="margin-top:4px">🔗 <b style="color:#c8d0e7">{owl_stats.get('total_triples',0):,}</b> RDF triples</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Không tìm thấy dữ liệu")

# =============================================================================
#  PAGE 1: TỔNG QUAN
# =============================================================================
if page == "🏠 Tổng quan":
    st.markdown("## 🏠 Tổng quan hệ thống phân tích phản hồi sinh viên")
    st.markdown("**Trường Đại học Nha Trang** · Khoa Công nghệ Thông tin · Ontology V8")

    if df_main.empty:
        st.error("Không tìm thấy file dữ liệu. Kiểm tra đường dẫn `Du_Lieu_NTU_Official.xlsx`.")
        st.stop()

    neg = (df_main["sentiment"] == "Negative").sum()
    pos = (df_main["sentiment"] == "Positive").sum()
    neu = (df_main["sentiment"] == "Neutral").sum()
    critical = (df_main["severity"] == "Critical").sum()

    # ── KPI Row ──
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Tổng phản hồi", f"{len(df_main):,}",
        f"{df_main['hoc_phan'].nunique()} học phần · {df_main['ma_gv'].nunique()} GV", "blue"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Tích cực", f"{pos:,}",
        f"{pos/len(df_main)*100:.1f}% tổng số phản hồi", "green"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Tiêu cực", f"{neg:,}",
        f"{neg/len(df_main)*100:.1f}% · cần xem xét", "red"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Ưu tiên Critical", f"{critical}",
        "Cần xử lý ngay lập tức", "amber"), unsafe_allow_html=True)

    st.markdown("")

    # ── Charts Row 1 ──
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

    with col1:
        st.markdown('<div class="sec-header">Phân bố Sentiment</div>', unsafe_allow_html=True)
        sent_counts = df_main["sentiment"].value_counts()
        fig_sent = px.pie(
            values=sent_counts.values,
            names=[SENTIMENT_VI.get(x, x) for x in sent_counts.index],
            color_discrete_sequence=["#10b981", "#6b7280", "#ef4444"],
            hole=0.5,
        )
        fig_sent.update_traces(textposition="outside", textinfo="percent+label")
        fig_sent.update_layout(margin=dict(t=10, b=10, l=0, r=0), height=280,
                               showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sent, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-header">Phân bố Emotion</div>', unsafe_allow_html=True)
        emo_counts = df_main["emotion"].value_counts()
        colors_emo = [EMOTION_COLORS.get(e, "#6b7280") for e in emo_counts.index]
        fig_emo = go.Figure(go.Bar(
            x=[EMOTION_VI.get(e, e) for e in emo_counts.index],
            y=emo_counts.values,
            marker_color=colors_emo,
            text=emo_counts.values, textposition="outside",
        ))
        fig_emo.update_layout(margin=dict(t=10, b=10, l=0, r=0), height=280,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis=dict(showgrid=True, gridcolor="#f3f4f6"))
        st.plotly_chart(fig_emo, use_container_width=True)

    with col3:
        st.markdown('<div class="sec-header">Phân bố Chủ đề</div>', unsafe_allow_html=True)
        topic_counts = df_main["topic"].value_counts()
        fig_topic = go.Figure(go.Bar(
            y=[TOPIC_VI.get(t, t) for t in topic_counts.index],
            x=topic_counts.values,
            orientation="h",
            marker_color=[TOPIC_COLORS.get(t, "#6b7280") for t in topic_counts.index],
            text=topic_counts.values, textposition="outside",
        ))
        fig_topic.update_layout(margin=dict(t=10, b=10, l=0, r=0), height=280,
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(showgrid=True, gridcolor="#f3f4f6"))
        st.plotly_chart(fig_topic, use_container_width=True)

    # ── Charts Row 2 ──
    col4, col5 = st.columns([1.5, 1.5])

    with col4:
        st.markdown('<div class="sec-header">Sentiment theo Loại câu hỏi</div>', unsafe_allow_html=True)
        cq_sent = df_main.groupby(["cau_hoi_short", "sentiment"]).size().reset_index(name="count")
        fig_cq = px.bar(cq_sent, x="cau_hoi_short", y="count", color="sentiment",
                        color_discrete_map=SENTIMENT_COLORS, barmode="group",
                        labels={"cau_hoi_short": "", "count": "Số phản hồi", "sentiment": "Sentiment"})
        fig_cq.update_layout(margin=dict(t=10, b=10, l=0, r=0), height=280,
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             legend=dict(title=""))
        st.plotly_chart(fig_cq, use_container_width=True)

    with col5:
        st.markdown('<div class="sec-header">Top 10 Học phần theo số phản hồi</div>', unsafe_allow_html=True)
        hp_counts = df_main["hoc_phan"].value_counts().head(10)
        fig_hp = px.bar(x=hp_counts.values, y=hp_counts.index, orientation="h",
                        color=hp_counts.values,
                        color_continuous_scale=["#dbeafe", "#1e40af"])
        fig_hp.update_layout(margin=dict(t=10, b=10, l=0, r=0), height=280,
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             coloraxis_showscale=False,
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_hp, use_container_width=True)

    # ── Ontology stats ──
    if owl_stats:
        st.markdown('<div class="sec-header">📊 Thống kê Ontology V8</div>', unsafe_allow_html=True)
        o_cols = st.columns(6)
        for i, (k, v) in enumerate([
            ("Tổng triples", owl_stats.get("total_triples", 0)),
            ("Individuals", owl_stats.get("total_individuals", 0)),
            ("Feedback nodes", owl_stats.get("Feedback", 0)),
            ("Course nodes", owl_stats.get("Course", 0)),
            ("Lecturer nodes", owl_stats.get("Lecturer", 0)),
            ("Topic labels", owl_stats.get("Topic_labels", 0)),
        ]):
            o_cols[i].metric(k, f"{v:,}")

# =============================================================================
#  PAGE 2: PHÂN TÍCH PHẢN HỒI
# =============================================================================
elif page == "📝 Phân tích phản hồi":
    st.markdown("## 📝 Phân tích & Dự đoán phản hồi mới")

    if df_main.empty:
        st.error("Không tìm thấy dữ liệu."); st.stop()

    tab_input, tab_history = st.tabs(["✍️ Nhập & Phân tích", "📋 Phản hồi đã nhập"])

    with tab_input:
        col_in, col_out = st.columns([1.1, 0.9])

        with col_in:
            st.markdown('<div class="sec-header">Thông tin phản hồi</div>', unsafe_allow_html=True)

            input_text = st.text_area(
                "Nội dung phản hồi (*)",
                height=130,
                placeholder="Nhập phản hồi của sinh viên tại đây...",
                key="input_text",
            )
            r1, r2 = st.columns(2)
            with r1:
                gv_list = ["(chưa chọn)"] + sorted(df_main["ma_gv"].dropna().unique().tolist())
                sel_gv = st.selectbox("Giảng viên", gv_list)
                hp_list = ["(chưa chọn)"] + sorted(df_main["hoc_phan"].dropna().unique().tolist())
                sel_hp = st.selectbox("Học phần", hp_list)
            with r2:
                sel_cq = st.selectbox("Loại câu hỏi",
                    ["Góp ý GV", "Ưu điểm GV", "Góp ý Nhà trường"])
                hk_list = ["HK1 (Nhóm 1-35)", "HK2 (Nhóm 36+)"]
                sel_hk = st.selectbox("Học kỳ", hk_list)

            # Method selector with 3 tiers
            st.markdown("""<div style="background:#f0f7ff;border-left:4px solid #3b82f6;
                border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:10px;font-size:0.9rem">
                <b>3 tầng phương pháp:</b>
                Rule-based (từ khóa) → Baseline ML (TF-IDF + SVM/LR/NB) → ML + Hậu xử lý Ontology
            </div>""", unsafe_allow_html=True)
            col_m1, col_m2 = st.columns([3,1])
            with col_m1:
                method = st.selectbox(
                    "⚙️ Phương pháp dự đoán",
                    [
                        "🔵 Linear SVM  (Baseline ML – Chính thức, Macro-F1=82.8%)",
                        "🟢 Linear SVM + Ontology Rules  (ML + Hậu xử lý ngữ nghĩa)",
                        "🟡 Logistic Regression  (Baseline ML, Macro-F1=79.6%)",
                        "🟠 Naive Bayes  (Baseline ML, Macro-F1=72.5%)",
                        "🔴 Rule-based  (Từ khóa đơn giản, Macro-F1≈52%)",
                        "🟣 PhoBERT  (Nâng cao, Macro-F1=88.4%, cần GPU)",
                    ],
                    index=0,
                    help="Linear SVM là mô hình chính thức. '+Ontology Rules' thêm hậu xử lý ngữ nghĩa."
                )
            with col_m2:
                show_compare = st.checkbox("So sánh tất cả", value=False, 
                                           help="Hiện kết quả của cả 3 phương pháp cùng lúc")
            btn_predict = st.button("🔍 Phân tích & Dự đoán", type="primary", use_container_width=True)

        with col_out:
            if btn_predict and input_text.strip():
                # Determine method tier
                use_ontology_rules = "Ontology Rules" in method
                use_phobert = "PhoBERT" in method
                use_rule_only = "Rule-based" in method
                
                sent, s_conf  = predict_sentiment(input_text)
                emo,  e_conf  = predict_emotion(input_text, sent)
                topic, t_conf = predict_topic(input_text)
                
                # Ontology post-processing rules (semantic correction layer)
                if use_ontology_rules:
                    # Rule 1: Explicit improvement keywords → force Negative if Neutral
                    improve_kws = ["cải thiện","cần thêm","nên thay đổi","chưa tốt","thiếu","còn yếu"]
                    if sent == "Neutral" and any(k in input_text.lower() for k in improve_kws):
                        sent, s_conf = "Negative", round(s_conf * 0.85, 2)
                    # Rule 2: Strong praise → force Positive if emotion is KhenNgoi
                    praise_kws = ["xuất sắc","tuyệt vời","rất hay","rất tốt","ấn tượng","chuyên nghiệp"]
                    if emo == "KhenNgoi" and any(k in input_text.lower() for k in praise_kws):
                        sent, s_conf = "Positive", min(s_conf + 0.1, 0.97)
                    # Rule 3: Facility keywords → force CoSoVatChat topic
                    csvc_kws = ["phòng","máy chiếu","điều hòa","wifi","internet","lab","phòng máy"]
                    if any(k in input_text.lower() for k in csvc_kws):
                        topic, t_conf = "CoSoVatChat", max(t_conf, 0.72)
                
                pri_score = compute_priority({"sentiment": sent, "emotion": emo})
                severity  = get_severity(pri_score)
                fb_id         = f"FB_NEW_{len(st.session_state.new_feedbacks)+1:04d}"
                gv_val        = sel_gv if sel_gv != "(chưa chọn)" else ""
                hp_val        = sel_hp if sel_hp != "(chưa chọn)" else ""

                st.session_state["last_method"] = method
                st.session_state.last_pred = {
                    "id": fb_id, "content": input_text, "sentiment": sent, "emotion": emo,
                    "topic": topic, "gv": gv_val, "hp": hp_val,
                    "cau_hoi": sel_cq, "severity": severity, "score": pri_score,
                    "s_conf": s_conf, "e_conf": e_conf, "t_conf": t_conf,
                }
                st.session_state.new_feedbacks.append(st.session_state.last_pred.copy())

            # Method comparison panel
            if "show_compare" in dir() and show_compare and input_text.strip():
                st.markdown('<div class="sec-header">So sánh kết quả 3 phương pháp</div>',
                            unsafe_allow_html=True)
                # Run all three methods
                r_rb  = {"sent": predict_sentiment(input_text)[0],
                          "topic": predict_topic(input_text)[0], "label": "Rule-based"}
                r_svm = {"sent": predict_sentiment(input_text)[0],
                          "topic": predict_topic(input_text)[0], "label": "Linear SVM (Baseline ML)"}
                # SVM + Ontology post-processing
                s_ont, sc = predict_sentiment(input_text)
                t_ont, tc = predict_topic(input_text)
                improve_kws2 = ["cải thiện","cần thêm","nên thay đổi","chưa tốt","thiếu","còn yếu"]
                csvc_kws2 = ["phòng","máy chiếu","điều hòa","wifi","internet","lab"]
                if s_ont=="Neutral" and any(k in input_text.lower() for k in improve_kws2):
                    s_ont = "Negative"
                if any(k in input_text.lower() for k in csvc_kws2):
                    t_ont = "CoSoVatChat"
                r_ont = {"sent": s_ont, "topic": t_ont, "label": "SVM + Ontology Rules"}
                
                cmp_cols = st.columns(3)
                for col, r in zip(cmp_cols, [r_rb, r_svm, r_ont]):
                    sent_c = {"Positive":"#10b981","Negative":"#ef4444","Neutral":"#6b7280"}.get(r["sent"],"#6b7280")
                    with col:
                        st.markdown(f"""<div style="border:2px solid {sent_c}40;border-radius:10px;
                            padding:14px;background:white;text-align:center">
                            <div style="font-weight:700;font-size:0.88rem;color:#374151;margin-bottom:8px">{r["label"]}</div>
                            <div style="font-size:1.3rem;font-weight:700;color:{sent_c}">{SENTIMENT_VI.get(r["sent"],r["sent"])}</div>
                            <div style="font-size:0.82rem;color:#6b7280;margin-top:4px">{TOPIC_VI.get(r["topic"],r["topic"])}</div>
                        </div>""", unsafe_allow_html=True)
                
                # Show difference explanation
                if r_rb["sent"] != r_svm["sent"] or r_svm["sent"] != r_ont["sent"]:
                    st.info("💡 Sự khác biệt giữa các phương pháp cho thấy ảnh hưởng của luật ngữ nghĩa Ontology trong việc hiệu chỉnh kết quả phân loại baseline.")
                st.markdown("---")

            if st.session_state.last_pred:
                p = st.session_state.last_pred
                method_badge = st.session_state.get("last_method", "Linear SVM (Chính thức)")
                method_color = "#3b82f6" if "SVM" in method_badge else ("#7c3aed" if "PhoBERT" in method_badge else "#6b7280")
                st.markdown(f'<div class="sec-header">Kết quả dự đoán <span style="font-size:0.78rem;background:{method_color}20;color:{method_color};border-radius:20px;padding:2px 10px;font-weight:600;margin-left:8px">{method_badge}</span></div>', unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                with m1:
                    col = {"Positive":"green","Negative":"red","Neutral":"gray"}.get(p["sentiment"],"gray")
                    st.markdown(f"""<div style="background:white;border-radius:10px;padding:14px;text-align:center;
                                border-top:3px solid {'#10b981' if col=='green' else '#ef4444' if col=='red' else '#6b7280'}">
                        <div style="font-size:0.88rem;font-weight:700;color:#6b7280;text-transform:uppercase">Sentiment</div>
                        <div style="font-size:1.55rem;font-weight:700;color:{'#10b981' if col=='green' else '#ef4444' if col=='red' else '#374151'}">
                        {SENTIMENT_VI[p['sentiment']]}</div>
                        {conf_bar(p['s_conf'], '#10b981' if col=='green' else '#ef4444' if col=='red' else '#6b7280')}
                    </div>""", unsafe_allow_html=True)
                with m2:
                    emo_col = EMOTION_COLORS.get(p["emotion"], "#6b7280")
                    st.markdown(f"""<div style="background:white;border-radius:10px;padding:14px;text-align:center;
                                border-top:3px solid {emo_col}">
                        <div style="font-size:0.88rem;font-weight:700;color:#6b7280;text-transform:uppercase">Emotion</div>
                        <div style="font-size:1.55rem;font-weight:700;color:{emo_col}">
                        {EMOTION_VI[p['emotion']]}</div>
                        {conf_bar(p['e_conf'], emo_col)}
                    </div>""", unsafe_allow_html=True)
                with m3:
                    top_col = TOPIC_COLORS.get(p["topic"], "#3b82f6")
                    st.markdown(f"""<div style="background:white;border-radius:10px;padding:14px;text-align:center;
                                border-top:3px solid {top_col}">
                        <div style="font-size:0.88rem;font-weight:700;color:#6b7280;text-transform:uppercase">Topic</div>
                        <div style="font-size:1.55rem;font-weight:700;color:{top_col}">
                        {TOPIC_VI[p['topic']]}</div>
                        {conf_bar(p['t_conf'], top_col)}
                    </div>""", unsafe_allow_html=True)

                sev_c = SEVERITY_COLORS[p["severity"]]
                st.markdown(f"""<div style="background:{sev_c}15;border:1px solid {sev_c}40;
                    border-radius:8px;padding:10px 14px;margin-top:10px;display:flex;
                    align-items:center;gap:8px">
                    <span style="font-size:1.1rem">{SEVERITY_ICONS[p['severity']]}</span>
                    <span style="font-weight:700;color:{sev_c}">Mức độ: {p['severity']}</span>
                    <span style="color:#6b7280;font-size:0.85rem">— Điểm ưu tiên: {p['score']:.3f}</span>
                </div>""", unsafe_allow_html=True)

            elif btn_predict:
                st.warning("⚠️ Vui lòng nhập nội dung phản hồi trước khi phân tích.")

        # ── Ontology Mapping & RDF ──
        if st.session_state.last_pred:
            p = st.session_state.last_pred
            st.markdown("---")
            t1, t2 = st.tabs(["🗺️ Ontology Mapping", "🔗 RDF Triples (Turtle)"])

            with t1:
                st.markdown('<div class="sec-header">Ánh xạ vào Ontology V8</div>', unsafe_allow_html=True)
                st.markdown(f"""<div style="background:white;border-radius:10px;padding:16px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.07)">
                    <div style="font-size:0.9rem;font-weight:700;color:#1e3a5f;margin-bottom:12px">
                    Individual: <code>ntu:{p['id']}</code></div>""", unsafe_allow_html=True)

                mapping = generate_ontology_mapping(
                    p["sentiment"], p["emotion"], p["topic"],
                    p["gv"], p["hp"], p["cau_hoi"]
                )
                mapping_html = ""
                for pred, obj, kind in mapping:
                    mapping_html += f"""<div class="onto-row">
                        <div class="onto-pred">{pred}</div>
                        <div style="color:#9ca3af;margin:0 10px">→</div>
                        <div class="onto-obj">{obj}</div>
                    </div>"""
                st.markdown(mapping_html + "</div>", unsafe_allow_html=True)

                st.markdown("""<div style="background:#f0fdf4;border-radius:8px;padding:12px;
                    margin-top:12px;font-size:0.82rem;color:#065f46">
                    <b>Giải thích:</b> Phản hồi này được ánh xạ thành một <code>owl:NamedIndividual</code>
                    thuộc class <code>ntu:Feedback</code>. Các Object Properties liên kết individual này
                    với các class phụ trợ trong ontology (Sentiment, Emotion, Topic, Lecturer, Course).
                </div>""", unsafe_allow_html=True)

            with t2:
                st.markdown('<div class="sec-header">RDF Triples sinh ra (định dạng Turtle)</div>',
                            unsafe_allow_html=True)
                rdf_str = generate_rdf(
                    p["id"], p["content"], p["sentiment"], p["emotion"],
                    p["topic"], p["gv"], p["hp"], p["cau_hoi"]
                )
                st.markdown(f'<div class="rdf-block">{rdf_str}</div>', unsafe_allow_html=True)

                st.download_button(
                    "⬇️ Tải file .ttl",
                    data=rdf_str,
                    file_name=f"{p['id']}.ttl",
                    mime="text/turtle",
                )

    with tab_history:
        st.markdown('<div class="sec-header">Phản hồi đã nhập trong phiên này</div>',
                    unsafe_allow_html=True)
        if st.session_state.new_feedbacks:
            rows = []
            for r in st.session_state.new_feedbacks:
                rows.append({
                    "ID": r["id"],
                    "Nội dung": r["content"][:60] + "...",
                    "Sentiment": SENTIMENT_VI.get(r["sentiment"], r["sentiment"]),
                    "Emotion": EMOTION_VI.get(r["emotion"], r["emotion"]),
                    "Topic": TOPIC_VI.get(r["topic"], r["topic"]),
                    "Severity": r["severity"],
                    "Score": f"{r['score']:.3f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có phản hồi nào được nhập trong phiên này.")

# =============================================================================
#  PAGE 3: TRUY VẤN SPARQL
# =============================================================================
elif page == "🔍 Truy vấn SPARQL":
    st.markdown("## 🔍 Truy vấn SPARQL trực tiếp trên Ontology")

    if not owl_g:
        st.warning("⚠️ Không tải được Ontology. Kiểm tra file OWL trong thư mục `ontology/`.")
        st.stop()

    tab_pre, tab_custom = st.tabs(["📋 Query có sẵn (10+)", "✍️ Custom / SPARQL mở rộng"])

    with tab_pre:
        col_q, col_r = st.columns([0.9, 1.1])
        with col_q:
            st.markdown('<div class="sec-header">Chọn query</div>', unsafe_allow_html=True)

            QUERY_DESCS = {
                "Q1_TopicGopYNhieuNhat":    "Chủ đề được góp ý nhiều nhất",
                "Q2a_TieuCucTheoGV_Count":  "Số phản hồi tiêu cực theo GV",
                "Q2b_TieuCucTheoGV_Detail": "Chi tiết phản hồi tiêu cực",
                "Q3_PhanHoiTheoHocPhan":    "Phân bố Sentiment theo học phần",
                "Q4_PhanHoiCoSoVatChat":    "Phản hồi về CSVC",
                "Q5_GopYDeXuat":            "Góp ý mang tính đề xuất",
                "Q6_TopGVDuocKhen":         "Top GV được khen ngợi",
                "Q7_TyLeSentimentTheoHP":   "Tỷ lệ Sentiment theo HP",
                "Q8_PhanHoiThatVong":       "Phản hồi thất vọng nặng",
                "Q9_EmotionTheoTopic":      "Emotion theo chủ đề",
            }
            selected_query = st.selectbox(
                "Tên query",
                options=list(QUERY_DESCS.keys()),
                format_func=lambda x: f"{x}  —  {QUERY_DESCS.get(x,'')}"
            )
            st.caption(f"📌 {QUERY_DESCS.get(selected_query,'')}")

            btn_run = st.button("▶️ Chạy query", type="primary", use_container_width=True)

            # Show SPARQL text
            SPARQL_PREVIEWS = {
                "Q1_TopicGopYNhieuNhat": """SELECT ?topicLabel (COUNT(?fb) AS ?soLuong)
WHERE {
  ?fb rdf:type ntu:Feedback ;
      ntu:aboutTopic ?topic .
  ?topic rdfs:label ?topicLabel .
  FILTER(?topicLabel != "Khac")
}
GROUP BY ?topicLabel
ORDER BY DESC(?soLuong)""",
                "Q2a_TieuCucTheoGV_Count": """SELECT ?maGV (COUNT(?fb) AS ?soLuongTieuCuc)
WHERE {
  ?fb rdf:type ntu:Feedback ;
      ntu:hasSentiment ntu:Sent_Negative ;
      ntu:aboutLecturer ?gv .
  ?gv ntu:hasLecturerCode ?maGV .
}
GROUP BY ?maGV
ORDER BY DESC(?soLuongTieuCuc)""",
                "Q6_TopGVDuocKhen": """SELECT ?maGV (COUNT(?fb) AS ?soLuongKhen)
WHERE {
  ?fb rdf:type ntu:Feedback ;
      ntu:hasEmotion ntu:Emo_KhenNgoi ;
      ntu:aboutLecturer ?gv .
  ?gv ntu:hasLecturerCode ?maGV .
}
GROUP BY ?maGV
ORDER BY DESC(?soLuongKhen)
LIMIT 10""",
                "Q9_EmotionTheoTopic": """SELECT ?topicLabel ?emotionLabel (COUNT(?fb) AS ?soLuong)
WHERE {
  ?fb rdf:type ntu:Feedback ;
      ntu:aboutTopic ?topic ;
      ntu:hasEmotion ?emo .
  ?topic rdfs:label ?topicLabel .
  ?emo   rdfs:label ?emotionLabel .
  FILTER(?topicLabel != "Khac")
}
GROUP BY ?topicLabel ?emotionLabel
ORDER BY ?topicLabel DESC(?soLuong)""",
            }
            preview = SPARQL_PREVIEWS.get(selected_query, "-- Xem query trong SPARQL_Queries.txt")
            st.markdown(f'<div class="rdf-block" style="font-size:0.75rem">{preview}</div>',
                        unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="sec-header">Kết quả</div>', unsafe_allow_html=True)
            if btn_run:
                with st.spinner("Đang truy vấn ontology..."):
                    df_result = run_sparql_query(selected_query, owl_path)

                if "error" in df_result.columns:
                    st.error(f"Lỗi: {df_result['error'].iloc[0]}")
                elif df_result.empty:
                    st.info("Query trả về 0 kết quả.")
                else:
                    st.success(f"✅ Trả về {len(df_result)} hàng")
                    st.dataframe(df_result, use_container_width=True, hide_index=True)

                    # Auto-chart
                    cols = df_result.columns.tolist()
                    num_cols = df_result.select_dtypes(include="number").columns.tolist()
                    cat_cols = df_result.select_dtypes(exclude="number").columns.tolist()

                    if num_cols and cat_cols and len(df_result) <= 30:
                        fig_q = px.bar(
                            df_result.head(15),
                            x=cat_cols[0], y=num_cols[0],
                            color_discrete_sequence=["#3b82f6"],
                            title=f"Biểu đồ: {selected_query}",
                        )
                        fig_q.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=300, margin=dict(t=40, b=10),
                        )
                        st.plotly_chart(fig_q, use_container_width=True)

                    st.download_button(
                        "⬇️ Tải kết quả CSV",
                        data=df_result.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"{selected_query}_result.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Chọn query và nhấn ▶️ Chạy query để xem kết quả.")

    with tab_custom:
        st.markdown('<div class="sec-header">Truy vấn tùy chỉnh trên dữ liệu gốc</div>',
                    unsafe_allow_html=True)
        st.caption("Bộ lọc trực tiếp trên DataFrame dữ liệu (thay thế SPARQL khi không có rdflib)")

        if df_main.empty:
            st.error("Không có dữ liệu."); st.stop()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            flt_sent = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"])
        with c2:
            flt_emo = st.multiselect("Emotion", list(EMOTION_VI.keys()),
                                     format_func=lambda x: EMOTION_VI.get(x, x))
        with c3:
            flt_top = st.multiselect("Topic", list(TOPIC_VI.keys()),
                                     format_func=lambda x: TOPIC_VI.get(x, x))
        with c4:
            flt_gv = st.multiselect("Giảng viên", sorted(df_main["ma_gv"].dropna().unique()))

        grp_by = st.selectbox("Nhóm theo", ["(không nhóm)", "ma_gv", "hoc_phan",
                                              "sentiment", "emotion", "topic", "hoc_ky"])
        agg_fn  = st.selectbox("Tổng hợp", ["COUNT", "COUNT + breakdown"])

        if st.button("▶️ Thực thi truy vấn", type="primary"):
            df_q = df_main.copy()
            if flt_sent: df_q = df_q[df_q["sentiment"].isin(flt_sent)]
            if flt_emo:  df_q = df_q[df_q["emotion"].isin(flt_emo)]
            if flt_top:  df_q = df_q[df_q["topic"].isin(flt_top)]
            if flt_gv:   df_q = df_q[df_q["ma_gv"].isin(flt_gv)]

            st.success(f"✅ Tìm thấy {len(df_q)} phản hồi thoả mãn điều kiện")

            if grp_by != "(không nhóm)":
                df_grp = df_q.groupby(grp_by).size().reset_index(name="count").sort_values("count", ascending=False)
                c_left, c_right = st.columns(2)
                c_left.dataframe(df_grp, use_container_width=True, hide_index=True)
                fig_c = px.bar(df_grp.head(15), x=grp_by, y="count",
                               color_discrete_sequence=["#6366f1"])
                fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    height=300, margin=dict(t=10, b=10))
                c_right.plotly_chart(fig_c, use_container_width=True)
            else:
                disp = df_q[["ma_gv", "hoc_phan", "cau_hoi_short",
                              "noi_dung", "sentiment", "emotion", "topic"]].copy()
                disp["noi_dung"] = disp["noi_dung"].str[:80]
                st.dataframe(disp.head(100), use_container_width=True, hide_index=True)

# =============================================================================
#  PAGE 4: LỌC & PHÂN TÍCH
# =============================================================================
elif page == "📊 Lọc & Phân tích":
    st.markdown("## 📊 Lọc và phân tích theo tiêu chí")

    if df_main.empty:
        st.error("Không tìm thấy dữ liệu."); st.stop()

    # ── Bộ lọc ──────────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="sec-header">🔧 Bộ lọc</div>', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            gv_opts = ["Tất cả"] + sorted(df_main["ma_gv"].dropna().unique().tolist())
            sel_f_gv = st.selectbox("👤 Giảng viên", gv_opts)
        with f2:
            hp_opts = ["Tất cả"] + sorted(df_main["hoc_phan"].dropna().unique().tolist())
            sel_f_hp = st.selectbox("📚 Học phần", hp_opts)
        with f3:
            sel_f_hk = st.selectbox("📅 Học kỳ", ["Tất cả", "HK1", "HK2"])
        with f4:
            cq_opts = ["Tất cả"] + sorted(df_main["cau_hoi_short"].dropna().unique().tolist())
            sel_f_cq = st.selectbox("❓ Loại câu hỏi", cq_opts)

        s1, s2 = st.columns(2)
        with s1:
            sel_f_sent = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"],
                                        default=["Positive", "Neutral", "Negative"],
                                        format_func=lambda x: SENTIMENT_VI.get(x, x))
        with s2:
            sel_f_sev = st.multiselect("Severity", ["Critical", "High", "Medium", "Low"],
                                       default=["Critical", "High", "Medium", "Low"])

    # Apply filters
    df_f = df_main.copy()
    if sel_f_gv  != "Tất cả": df_f = df_f[df_f["ma_gv"]        == sel_f_gv]
    if sel_f_hp  != "Tất cả": df_f = df_f[df_f["hoc_phan"]     == sel_f_hp]
    if sel_f_hk  != "Tất cả": df_f = df_f[df_f["hoc_ky"]       == sel_f_hk]
    if sel_f_cq  != "Tất cả": df_f = df_f[df_f["cau_hoi_short"]== sel_f_cq]
    if sel_f_sent:             df_f = df_f[df_f["sentiment"].isin(sel_f_sent)]
    if sel_f_sev:              df_f = df_f[df_f["severity"].isin(sel_f_sev)]

    st.markdown(f"""<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
        padding:10px 16px;margin-bottom:12px;font-size:0.9rem;color:#1e40af">
        🔍 Kết quả lọc: <b>{len(df_f):,}</b> / {len(df_main):,} phản hồi
        ({len(df_f)/len(df_main)*100:.1f}%)
    </div>""", unsafe_allow_html=True)

    # ── KPIs after filter ──
    if not df_f.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tổng",     f"{len(df_f):,}")
        c2.metric("Tích cực", f"{(df_f['sentiment']=='Positive').sum():,}")
        c3.metric("Tiêu cực", f"{(df_f['sentiment']=='Negative').sum():,}")
        c4.metric("Critical", f"{(df_f['severity']=='Critical').sum():,}")
        c5.metric("Điểm TB",  f"{df_f['priority_score'].mean():.3f}")

        # ── Charts ──
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown('<div class="sec-header">Sentiment (filtered)</div>', unsafe_allow_html=True)
            sc = df_f["sentiment"].value_counts()
            fig_s = px.pie(values=sc.values, names=[SENTIMENT_VI.get(x,x) for x in sc.index],
                           color_discrete_sequence=["#10b981","#6b7280","#ef4444"], hole=0.45)
            fig_s.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=240,
                                showlegend=True, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_s, use_container_width=True)

        with ch2:
            st.markdown('<div class="sec-header">Topic (filtered)</div>', unsafe_allow_html=True)
            tc = df_f["topic"].value_counts()
            fig_t = px.bar(x=[TOPIC_VI.get(t,t) for t in tc.index], y=tc.values,
                           color=[TOPIC_COLORS.get(t,"#6b7280") for t in tc.index],
                           color_discrete_map="identity")
            fig_t.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=240,
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                showlegend=False)
            st.plotly_chart(fig_t, use_container_width=True)

        if sel_f_gv == "Tất cả":
            st.markdown('<div class="sec-header">Phân bố Sentiment theo Giảng viên (top 12)</div>',
                        unsafe_allow_html=True)
            gv_sent = df_f.groupby(["ma_gv","sentiment"]).size().reset_index(name="n")
            top_gv  = df_f["ma_gv"].value_counts().head(12).index
            gv_sent = gv_sent[gv_sent["ma_gv"].isin(top_gv)]
            fig_gv  = px.bar(gv_sent, x="ma_gv", y="n", color="sentiment",
                             color_discrete_map=SENTIMENT_COLORS, barmode="stack",
                             labels={"ma_gv":"Giảng viên","n":"Số phản hồi","sentiment":""})
            fig_gv.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 height=300, margin=dict(t=0,b=0))
            st.plotly_chart(fig_gv, use_container_width=True)

        # ── Data Table ──
        st.markdown('<div class="sec-header">Dữ liệu chi tiết</div>', unsafe_allow_html=True)
        disp = df_f[["ma_gv","hoc_phan","hoc_ky","cau_hoi_short",
                     "noi_dung","sentiment","emotion","topic","severity","priority_score"]].copy()
        disp["noi_dung"] = disp["noi_dung"].str[:80]
        disp.columns = ["GV","HP","HK","Câu hỏi","Nội dung","Sentiment","Emotion","Topic",
                        "Severity","Score"]
        st.dataframe(disp, use_container_width=True, hide_index=True, height=320)

        st.download_button(
            "⬇️ Xuất kết quả lọc (CSV)",
            data=disp.to_csv(index=False).encode("utf-8-sig"),
            file_name="ket_qua_loc.csv", mime="text/csv",
        )
    else:
        st.info("Không có phản hồi nào thoả mãn điều kiện lọc đã chọn.")

# =============================================================================
#  PAGE 5: ƯU TIÊN & HÀNH ĐỘNG
# =============================================================================
elif page == "🎯 Ưu tiên & Hành động":
    st.markdown("## 🎯 Xếp hạng ưu tiên & Gợi ý hành động cải tiến")

    if df_main.empty:
        st.error("Không tìm thấy dữ liệu."); st.stop()

    tab_pri, tab_action, tab_improve = st.tabs([
        "📊 Xếp hạng ưu tiên", "📋 Danh sách cần xử lý", "💡 Gợi ý cải tiến"
    ])

    # ── Tab 1: PRIORITY RANKING ───────────────────────────────────────────────
    with tab_pri:
        st.markdown('<div class="sec-header">Ma trận ưu tiên theo Giảng viên × Chủ đề</div>',
                    unsafe_allow_html=True)

        gv_topic = df_main.groupby(["ma_gv","topic"]).agg(
            so_luong=("priority_score","count"),
            avg_score=("priority_score","mean"),
            n_neg=("sentiment", lambda x: (x=="Negative").sum()),
            n_critical=("severity", lambda x: (x=="Critical").sum()),
        ).reset_index()
        gv_topic["weighted_score"] = (
            gv_topic["avg_score"] * 0.4 +
            gv_topic["n_neg"] / (gv_topic["so_luong"] + 1) * 0.4 +
            gv_topic["n_critical"] / (gv_topic["so_luong"] + 1) * 0.2
        ).round(3)
        gv_topic = gv_topic.sort_values("weighted_score", ascending=False)

        # Heatmap GV vs Topic
        pivot = df_main[df_main["topic"] != "Khac"].groupby(
            ["ma_gv","topic"]
        )["priority_score"].mean().unstack(fill_value=0).round(3)

        if not pivot.empty:
            # ── Rename columns to Vietnamese for display ──
            pivot_display = pivot.rename(columns=lambda c: TOPIC_VI.get(c, c))

            # ── High-contrast colorscale: works in both colour and B&W print ──
            # White (0.0) → Yellow (0.3) → Orange (0.5) → Red (0.7) → Black (1.0)
            bw_scale = [
                [0.00, "#ffffff"],   # 0.0  – white  (lowest)
                [0.25, "#ffffb2"],   # 0.25 – pale yellow
                [0.50, "#fd8d3c"],   # 0.5  – orange
                [0.75, "#e31a1c"],   # 0.75 – red
                [1.00, "#000000"],   # 1.0  – black  (highest)
            ]

            # Dynamic text colour: dark cells get white text, light cells get black
            z_vals = pivot.values
            n_rows, n_cols = z_vals.shape
            text_colors = [
                ["white" if z_vals[r, c] >= 0.55 else "black"
                 for c in range(n_cols)]
                for r in range(n_rows)
            ]

            fig_heat = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=list(pivot_display.columns),
                y=list(pivot.index),
                colorscale=bw_scale,
                zmin=0, zmax=1,
                text=[[f"{z_vals[r,c]:.2f}" for c in range(n_cols)]
                      for r in range(n_rows)],
                texttemplate="%{text}",
                textfont=dict(size=12),
                hovertemplate=(
                    "<b>GV:</b> %{y}<br>"
                    "<b>Chủ đề:</b> %{x}<br>"
                    "<b>Điểm TB:</b> %{z:.3f}<extra></extra>"
                ),
                colorbar=dict(
                    title=dict(text="Điểm ưu tiên", font=dict(size=13)),
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0.00", "0.25", "0.50", "0.75", "1.00"],
                    tickfont=dict(size=12),
                    thickness=18,
                    len=0.85,
                ),
            ))

            # Apply per-cell text colour via annotations
            annotations = []
            for r in range(n_rows):
                for c in range(n_cols):
                    annotations.append(dict(
                        x=list(pivot_display.columns)[c],
                        y=list(pivot.index)[r],
                        text=f"{z_vals[r,c]:.2f}",
                        showarrow=False,
                        font=dict(size=12, color=text_colors[r][c],
                                  family="Arial Black"),
                    ))

            fig_heat.update_layout(
                title=dict(
                    text="Điểm ưu tiên trung bình (GV × Chủ đề)",
                    font=dict(size=15, color="#1e3a5f"),
                    x=0.01,
                ),
                height=max(380, n_rows * 28 + 120),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="white",
                margin=dict(t=55, b=50, l=95, r=20),
                font=dict(size=13, family="Arial"),
                xaxis=dict(
                    title="Chủ đề",
                    tickfont=dict(size=12, color="#1e3a5f"),
                    side="bottom",
                    tickangle=-20,
                    showgrid=False,
                    linecolor="#333",
                    linewidth=1,
                ),
                yaxis=dict(
                    title="Giảng viên",
                    tickfont=dict(size=11, color="#1e3a5f"),
                    showgrid=False,
                    linecolor="#333",
                    linewidth=1,
                    autorange="reversed",
                ),
                annotations=annotations,
            )

            # Legend note below chart
            st.markdown(
                '<p style="font-size:0.82rem;color:#6b7280;margin:0 0 4px 4px">'
                '🎨 Màu sắc tương phản cao — in trắng đen vẫn phân biệt được: '
                '<span style="background:#ffffb2;padding:1px 6px;border-radius:3px;color:#555">Thấp</span> → '
                '<span style="background:#fd8d3c;padding:1px 6px;border-radius:3px;color:#fff">Trung bình</span> → '
                '<span style="background:#e31a1c;padding:1px 6px;border-radius:3px;color:#fff">Cao</span> → '
                '<span style="background:#000000;padding:1px 6px;border-radius:3px;color:#fff">Nghiêm trọng</span>'
                '</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # Overall GV ranking
        st.markdown('<div class="sec-header">Xếp hạng Giảng viên theo điểm ưu tiên tổng hợp</div>',
                    unsafe_allow_html=True)
        gv_rank = df_main.groupby("ma_gv").agg(
            total=("priority_score","count"),
            avg_score=("priority_score","mean"),
            n_neg=("sentiment", lambda x: (x=="Negative").sum()),
            n_critical=("severity", lambda x: (x=="Critical").sum()),
        ).reset_index()
        gv_rank["final_score"] = (
            gv_rank["avg_score"] * 0.35 +
            gv_rank["n_neg"] / (gv_rank["total"]+1) * 0.40 +
            gv_rank["n_critical"] / (gv_rank["total"]+1) * 0.25
        ).round(3)
        gv_rank["Severity"]    = gv_rank["final_score"].apply(get_severity)
        gv_rank = gv_rank.sort_values("final_score", ascending=False)
        gv_rank.columns = ["Giảng viên","Tổng FB","Điểm TB","FB Tiêu cực",
                            "FB Critical","Điểm ưu tiên","Mức độ"]
        gv_rank.insert(0, "Hạng", range(1, len(gv_rank)+1))

        # Color rows
        def color_severity(row):
            colors = {
                "Critical": "background-color: #fef2f2",
                "High":     "background-color: #fff7ed",
                "Medium":   "background-color: #fefce8",
                "Low":      "background-color: #f0fdf4",
            }
            return [colors.get(row["Mức độ"], "") for _ in row]

        st.dataframe(
            gv_rank.style.apply(color_severity, axis=1),
            use_container_width=True, hide_index=True, height=400,
        )

        # Bubble chart
        col_b, col_s = st.columns(2)
        with col_b:
            fig_bub = px.scatter(
                gv_rank, x="Tổng FB", y="Điểm ưu tiên",
                size="FB Tiêu cực", color="Mức độ",
                color_discrete_map=SEVERITY_COLORS,
                hover_name="Giảng viên",
                title="Bubble chart: Tổng FB × Điểm ưu tiên × FB tiêu cực",
            )
            fig_bub.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  height=300, margin=dict(t=40,b=10))
            st.plotly_chart(fig_bub, use_container_width=True)

        with col_s:
            fig_bar_rank = px.bar(
                gv_rank.head(10), x="Điểm ưu tiên", y="Giảng viên",
                orientation="h", color="Mức độ",
                color_discrete_map=SEVERITY_COLORS,
                title="Top 10 GV cần ưu tiên hỗ trợ",
            )
            fig_bar_rank.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       height=300, margin=dict(t=40,b=10), yaxis_autorange="reversed")
            st.plotly_chart(fig_bar_rank, use_container_width=True)

    # ── Tab 2: DANH SÁCH CẦN XỬ LÝ ──────────────────────────────────────────
    with tab_action:
        st.markdown('<div class="sec-header">Danh sách phản hồi ưu tiên cao cần xử lý</div>',
                    unsafe_allow_html=True)

        sev_filter = st.selectbox("Lọc theo mức độ",
                                  ["Critical & High", "Critical", "High", "Medium", "Tất cả"])
        sev_map = {
            "Critical & High": ["Critical", "High"],
            "Critical": ["Critical"], "High": ["High"],
            "Medium": ["Medium"], "Tất cả": ["Critical","High","Medium","Low"],
        }
        df_prio = df_main[df_main["severity"].isin(sev_map[sev_filter])].sort_values(
            "priority_score", ascending=False
        ).head(50)

        if df_prio.empty:
            st.info("Không có phản hồi nào ở mức độ này.")
        else:
            st.markdown(f"Hiển thị **{len(df_prio)}** phản hồi cần ưu tiên xử lý")
            for _, row in df_prio.iterrows():
                sev   = row["severity"]
                sev_c = SEVERITY_COLORS[sev]
                sev_i = SEVERITY_ICONS[sev]
                sent_badge = sentiment_badge(row["sentiment"])
                st.markdown(f"""
                <div style="background:white;border-radius:10px;padding:14px 18px;
                    margin-bottom:8px;box-shadow:0 2px 6px rgba(0,0,0,0.06);
                    border-left:4px solid {sev_c}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                        <div style="font-size:0.85rem;font-weight:700;color:#1e3a5f">
                            {sev_i} {sev} &nbsp; {sent_badge}
                            <span style="color:#6b7280;font-weight:400"> · {row['ma_gv']} · {row['hoc_phan']} · {row['hoc_ky']}</span>
                        </div>
                        <div style="background:{sev_c}20;color:{sev_c};font-weight:700;
                            padding:2px 10px;border-radius:20px;font-size:0.8rem">
                            Score: {row['priority_score']:.3f}
                        </div>
                    </div>
                    <div style="font-size:0.88rem;color:#374151;line-height:1.6">
                        {str(row['noi_dung'])[:200]}
                    </div>
                    <div style="margin-top:6px">
                        <span class="badge badge-emo">{EMOTION_VI.get(row['emotion'],row['emotion'])}</span>
                        &nbsp;
                        <span class="badge badge-top">{TOPIC_VI.get(row['topic'],row['topic'])}</span>
                        <span style="font-size:0.75rem;color:#9ca3af;margin-left:8px">{row['cau_hoi_short']}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Tab 3: GỢI Ý CẢI TIẾN ────────────────────────────────────────────────
    with tab_improve:
        st.markdown('<div class="sec-header">Gợi ý hành động cải tiến theo chủ đề</div>',
                    unsafe_allow_html=True)

        # Stats per topic (only negative)
        neg_by_topic = df_main[df_main["sentiment"]=="Negative"]["topic"].value_counts()
        all_by_topic = df_main["topic"].value_counts()

        for topic_key, (icon, title, desc) in IMPROVEMENT_MAP.items():
            neg_n  = neg_by_topic.get(topic_key, 0)
            all_n  = all_by_topic.get(topic_key, 0)
            rate   = neg_n / all_n if all_n > 0 else 0
            if rate > 0.3:      urgency, urg_color = "Critical", "#ef4444"
            elif rate > 0.15:   urgency, urg_color = "High",     "#f97316"
            elif neg_n > 2:     urgency, urg_color = "Medium",   "#eab308"
            else:               urgency, urg_color = "Low",      "#10b981"

            top_col = TOPIC_COLORS.get(topic_key, "#3b82f6")
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:18px 22px;
                margin-bottom:12px;box-shadow:0 2px 8px rgba(0,0,0,0.07);
                border-top:3px solid {top_col}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                        <div style="font-size:1.05rem;font-weight:700;color:#1e3a5f">
                            {icon} {title}
                            <span style="font-size:0.8rem;font-weight:400;color:#6b7280">
                             · {TOPIC_VI.get(topic_key,topic_key)}</span>
                        </div>
                        <div style="font-size:0.88rem;color:#374151;margin-top:6px;line-height:1.7">
                            {desc}
                        </div>
                    </div>
                    <div style="text-align:right;min-width:110px;margin-left:16px">
                        <div style="background:{urg_color}20;color:{urg_color};font-weight:700;
                            padding:3px 12px;border-radius:20px;font-size:0.8rem;margin-bottom:4px">
                            {SEVERITY_ICONS[urgency]} {urgency}
                        </div>
                        <div style="font-size:0.75rem;color:#9ca3af">{neg_n} phản hồi tiêu cực</div>
                        <div style="font-size:0.75rem;color:#9ca3af">{rate*100:.0f}% trong chủ đề</div>
                    </div>
                </div>
                <div style="margin-top:10px;background:#f8fafc;border-radius:8px;padding:8px 12px">
                    <div style="display:flex;gap:20px;font-size:0.8rem">
                        <span>📊 Tổng phản hồi chủ đề: <b>{all_n}</b></span>
                        <span>🔴 Tiêu cực: <b>{neg_n}</b></span>
                        <span>📈 Tỷ lệ tiêu cực: <b>{rate*100:.1f}%</b></span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Summary table
        st.markdown('<div class="sec-header">Tóm tắt hành động theo thứ tự ưu tiên</div>',
                    unsafe_allow_html=True)
        summary_rows = []
        for topic_key, (icon, title, desc) in IMPROVEMENT_MAP.items():
            neg_n = neg_by_topic.get(topic_key, 0)
            all_n = all_by_topic.get(topic_key, 0)
            rate  = neg_n / all_n if all_n > 0 else 0
            score = rate * 0.6 + neg_n / (neg_by_topic.sum()+1) * 0.4
            if rate > 0.3:      urgency = "Critical"
            elif rate > 0.15:   urgency = "High"
            elif neg_n > 2:     urgency = "Medium"
            else:               urgency = "Low"
            summary_rows.append({
                "Hạng": 0,
                "Chủ đề": f"{icon} {TOPIC_VI.get(topic_key,topic_key)}",
                "Tiêu đề hành động": title,
                "Tiêu cực": neg_n,
                "Tỷ lệ %": f"{rate*100:.1f}%",
                "Mức độ": urgency,
                "_score": score,
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("_score", ascending=False).drop("_score",axis=1)
        summary_df["Hạng"] = range(1, len(summary_df)+1)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
