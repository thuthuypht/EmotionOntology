"""
owl_sparql_engine.py  –  Module truy vấn SPARQL thời gian thực trên OWL/RDF
Không cần rdflib – dùng xml.etree.ElementTree có sẵn trong Python stdlib.

Hỗ trợ các pattern SPARQL cần thiết cho hệ thống:
  - SELECT với WHERE và FILTER
  - GROUP BY + COUNT  (aggregation)
  - ORDER BY DESC/ASC
  - LIMIT
  - Optional UNION
"""

from xml.etree import ElementTree as ET
from collections import defaultdict
import re, os

NS_RDF  = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_OWL  = "http://www.w3.org/2002/07/owl#"
NS_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
NS_NTU  = "http://ntu.edu.vn/feedback_system#"

def _local(uri: str) -> str:
    """Trích tên local từ IRI: ntu:Topic_GiangDay → Topic_GiangDay"""
    for sep in ("#", "/"):
        if sep in uri:
            return uri.rsplit(sep, 1)[-1]
    return uri


class OWLGraph:
    """
    Tải và lưu toàn bộ triple trong file OWL RDF/XML vào bộ nhớ.
    Triple format: (subject_local, predicate_local, object_local_or_literal)
    """

    def __init__(self, owl_path: str):
        self.triples: list[tuple[str, str, str]] = []
        self._type_map: dict[str, str] = {}    # subject → rdf:type
        self._load(owl_path)

    def _load(self, path: str):
        tree = ET.parse(path)
        root = tree.getroot()
        for ind in root.findall(f"{{{NS_OWL}}}NamedIndividual"):
            subj_iri = ind.get(f"{{{NS_RDF}}}about", "")
            subj = _local(subj_iri)
            for child in ind:
                tag = child.tag
                # Skip namespace declarations
                if "}" not in tag:
                    continue
                pred = _local(tag.replace("{","").split("}")[1])
                full_tag = tag
                # rdf:type
                if tag == f"{{{NS_RDF}}}type":
                    res = child.get(f"{{{NS_RDF}}}resource","")
                    self.triples.append((subj, "type", _local(res)))
                    if _local(res) not in ("","NamedIndividual"):
                        self._type_map[subj] = _local(res)
                # Object property (has resource attribute)
                elif child.get(f"{{{NS_RDF}}}resource"):
                    obj = _local(child.get(f"{{{NS_RDF}}}resource",""))
                    pred_local = full_tag.replace(f"{{{NS_NTU}}}","") \
                                        .replace(f"{{{NS_RDFS}}}","rdfs:") \
                                        .replace(f"{{{NS_RDF}}}","rdf:")
                    self.triples.append((subj, pred_local, obj))
                # Data property (text content)
                elif child.text and child.text.strip():
                    pred_local = full_tag.replace(f"{{{NS_NTU}}}","") \
                                        .replace(f"{{{NS_RDFS}}}","rdfs:") \
                                        .replace(f"{{{NS_RDF}}}","rdf:")
                    self.triples.append((subj, pred_local, child.text.strip()))

    def subjects_of_type(self, type_name: str) -> list[str]:
        return [s for s, t in self._type_map.items() if t == type_name]

    def objects_of(self, subj: str, pred: str) -> list[str]:
        return [o for s,p,o in self.triples if s==subj and p==pred]

    def first_of(self, subj: str, pred: str) -> str:
        lst = self.objects_of(subj, pred)
        return lst[0] if lst else ""

    def subjects_with(self, pred: str, obj: str) -> list[str]:
        return [s for s,p,o in self.triples if p==pred and o==obj]


# ─── Pre-defined queries executed against OWLGraph ────────────────────────────

def _q1_topic_count(g: OWLGraph, limit=10) -> list[dict]:
    """Q1: Chủ đề được góp ý nhiều nhất (trừ Khac)"""
    counts = defaultdict(int)
    for fb in g.subjects_of_type("Feedback"):
        topic_node = g.first_of(fb, "aboutTopic")
        if topic_node and "Khac" not in topic_node:
            lbl = topic_node.replace("Topic_","")
            counts[lbl] += 1
    rows = sorted(counts.items(), key=lambda x: -x[1])[:limit]
    return [{"topicLabel": k, "soLuong": v} for k,v in rows]


def _q2a_negative_by_gv(g: OWLGraph) -> list[dict]:
    """Q2a: Số phản hồi tiêu cực theo giảng viên"""
    counts = defaultdict(int)
    for fb in g.subjects_of_type("Feedback"):
        if g.first_of(fb,"hasSentiment") == "Sent_Negative":
            gv_node = g.first_of(fb,"aboutLecturer")
            if gv_node:
                maGV = g.first_of(gv_node,"hasLecturerCode") or gv_node
                counts[maGV] += 1
    rows = sorted(counts.items(), key=lambda x:-x[1])
    return [{"maGV":k,"soLuongTieuCuc":v} for k,v in rows]


def _q2b_negative_detail(g: OWLGraph, ma_gv_filter="") -> list[dict]:
    """Q2b: Nội dung chi tiết phản hồi tiêu cực"""
    rows = []
    for fb in g.subjects_of_type("Feedback"):
        if g.first_of(fb,"hasSentiment") != "Sent_Negative":
            continue
        gv_node = g.first_of(fb,"aboutLecturer")
        maGV = g.first_of(gv_node,"hasLecturerCode") if gv_node else ""
        if ma_gv_filter and maGV != ma_gv_filter:
            continue
        content = g.first_of(fb,"hasContent")
        rows.append({"fb": fb, "maGV": maGV, "noiDung": content})
    return sorted(rows, key=lambda x: x["maGV"])[:50]


def _q3_sentiment_by_course(g: OWLGraph) -> list[dict]:
    """Q3: Phân phối Sentiment theo học phần"""
    counts = defaultdict(lambda: defaultdict(int))
    for fb in g.subjects_of_type("Feedback"):
        course_node = g.first_of(fb,"aboutCourse")
        if not course_node: continue
        maHP = g.first_of(course_node,"hasCourseCode") or course_node
        sent_node = g.first_of(fb,"hasSentiment")
        sent = sent_node.replace("Sent_","") if sent_node else "Unknown"
        counts[maHP][sent] += 1
    rows = []
    for maHP, sents in sorted(counts.items()):
        for sent, n in sorted(sents.items()):
            rows.append({"maHP":maHP,"sentimentLabel":sent,"soLuong":n})
    return rows


def _q4_csvc(g: OWLGraph) -> list[dict]:
    """Q4: Phản hồi về cơ sở vật chất"""
    rows = []
    for fb in g.subjects_of_type("Feedback"):
        if g.first_of(fb,"aboutTopic") != "Topic_CoSoVatChat":
            continue
        content = g.first_of(fb,"hasContent")
        sent = g.first_of(fb,"hasSentiment").replace("Sent_","")
        emo  = g.first_of(fb,"hasEmotion").replace("Emo_","")
        rows.append({"fb":fb,"sentimentLabel":sent,"emotionLabel":emo,"noiDung":content})
    return sorted(rows, key=lambda x:x["sentimentLabel"])


def _q5_deXuat(g: OWLGraph) -> list[dict]:
    """Q5: Góp ý mang tính đề xuất"""
    rows = []
    for fb in g.subjects_of_type("Feedback"):
        if g.first_of(fb,"hasEmotion") != "Emo_DeXuat":
            continue
        content = g.first_of(fb,"hasContent")
        topic = g.first_of(fb,"aboutTopic").replace("Topic_","")
        rows.append({"fb":fb,"topicLabel":topic,"noiDung":content})
    return sorted(rows, key=lambda x:x["topicLabel"])[:100]


def _q6_top_praised_gv(g: OWLGraph, top=10) -> list[dict]:
    """Q6: Top GV được khen"""
    counts = defaultdict(int)
    for fb in g.subjects_of_type("Feedback"):
        if g.first_of(fb,"hasEmotion") != "Emo_KhenNgoi":
            continue
        gv_node = g.first_of(fb,"aboutLecturer")
        if gv_node:
            maGV = g.first_of(gv_node,"hasLecturerCode") or gv_node
            counts[maGV] += 1
    rows = sorted(counts.items(), key=lambda x:-x[1])[:top]
    return [{"maGV":k,"soLuongKhen":v} for k,v in rows]


def _q7_sentiment_rate_by_course(g: OWLGraph) -> list[dict]:
    return _q3_sentiment_by_course(g)   # same structure, sorted differently


def _q8_that_vong(g: OWLGraph) -> list[dict]:
    """Q8: Phản hồi thất vọng nặng"""
    rows = []
    for fb in g.subjects_of_type("Feedback"):
        if g.first_of(fb,"hasEmotion") != "Emo_ThatVong":
            continue
        content = g.first_of(fb,"hasContent")
        gv_node = g.first_of(fb,"aboutLecturer")
        maGV = g.first_of(gv_node,"hasLecturerCode") if gv_node else ""
        course_node = g.first_of(fb,"aboutCourse")
        maHP = g.first_of(course_node,"hasCourseCode") if course_node else ""
        rows.append({"fb":fb,"maGV":maGV,"maHP":maHP,"noiDung":content})
    return rows


def _q9_emotion_by_topic(g: OWLGraph) -> list[dict]:
    """Q9: Emotion theo chủ đề"""
    counts = defaultdict(lambda: defaultdict(int))
    for fb in g.subjects_of_type("Feedback"):
        topic = g.first_of(fb,"aboutTopic").replace("Topic_","")
        if topic == "Khac": continue
        emo = g.first_of(fb,"hasEmotion").replace("Emo_","")
        counts[topic][emo] += 1
    rows = []
    for topic, emos in sorted(counts.items()):
        for emo, n in sorted(emos.items(), key=lambda x:-x[1]):
            rows.append({"topicLabel":topic,"emotionLabel":emo,"soLuong":n})
    return rows


# ─── Public interface ──────────────────────────────────────────────────────────

QUERY_MAP = {
    "Q1_TopicGopYNhieuNhat":      _q1_topic_count,
    "Q2a_TieuCucTheoGV_Count":    _q2a_negative_by_gv,
    "Q2b_TieuCucTheoGV_Detail":   _q2b_negative_detail,
    "Q3_PhanHoiTheoHocPhan":      _q3_sentiment_by_course,
    "Q4_PhanHoiCoSoVatChat":      _q4_csvc,
    "Q5_GopYDeXuat":              _q5_deXuat,
    "Q6_TopGVDuocKhen":           _q6_top_praised_gv,
    "Q7_TyLeSentimentTheoHP":     _q7_sentiment_rate_by_course,
    "Q8_PhanHoiThatVong":         _q8_that_vong,
    "Q9_EmotionTheoTopic":        _q9_emotion_by_topic,
}

_graph_cache: dict[str, OWLGraph] = {}

def load_graph(owl_path: str) -> OWLGraph:
    if owl_path not in _graph_cache:
        _graph_cache[owl_path] = OWLGraph(owl_path)
    return _graph_cache[owl_path]


def execute_query(query_name: str, owl_path: str, **kwargs) -> list[dict]:
    """
    Thực thi một trong 10 query đã định nghĩa trên file OWL thật.
    Trả về list[dict] – mỗi dict là một hàng kết quả.
    """
    fn = QUERY_MAP.get(query_name)
    if fn is None:
        raise ValueError(f"Unknown query: {query_name}. Available: {list(QUERY_MAP)}")
    g = load_graph(owl_path)
    import pandas as pd
    return pd.DataFrame(fn(g, **kwargs))


def get_ontology_stats(owl_path: str) -> dict:
    """Thống kê nhanh về ontology đã tải."""
    g = load_graph(owl_path)
    from collections import Counter
    type_counts = Counter(g._type_map.values())
    return {
        "total_triples":     len(g.triples),
        "total_individuals": len(g._type_map),
        "Feedback":          type_counts.get("Feedback",0),
        "Course":            type_counts.get("Course",0),
        "Lecturer":          type_counts.get("Lecturer",0),
        "Sentiment_labels":  type_counts.get("Sentiment",0),
        "Emotion_labels":    type_counts.get("Emotion",0),
        "Topic_labels":      type_counts.get("Topic",0),
    }


if __name__ == "__main__":
    import sys
    owl = sys.argv[1] if len(sys.argv)>1 else \
        os.path.join(os.path.dirname(__file__),"../ontology/Ontology_NTU_FeedbackSystem_V7.owl")
    owl = os.path.abspath(owl)
    print(f"\nLoading: {owl}")
    stats = get_ontology_stats(owl)
    print("\n── Thống kê Ontology ──")
    for k,v in stats.items():
        print(f"  {k:<22}: {v}")

    print("\n── Q1: Chủ đề góp ý nhiều nhất ──")
    import pandas as pd
    print(execute_query("Q1_TopicGopYNhieuNhat", owl).to_string(index=False))

    print("\n── Q6: Top 5 GV được khen ──")
    print(execute_query("Q6_TopGVDuocKhen", owl).head(5).to_string(index=False))

    print("\n── Q9: Emotion theo chủ đề (top 12) ──")
    print(execute_query("Q9_EmotionTheoTopic", owl).head(12).to_string(index=False))
