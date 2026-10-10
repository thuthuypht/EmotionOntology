"""
=============================================================================
SPARQL QUERIES V8 — Minh họa các thành phần mở rộng
=============================================================================
Gồm 12 queries:
  Q_V8_01 → Q_V8_04 : khai thác SeverityLevel
  Q_V8_05 → Q_V8_07 : khai thác Recommendation + hasPriorityScore
  Q_V8_08 → Q_V8_09 : khai thác Department / Faculty
  Q_V8_10           : khai thác AcademicTerm / belongsToAcademicTerm
  Q_V8_11           : khai thác FeedbackSource / DataOrigin
  Q_V8_12           : query tổng hợp đa chiều
=============================================================================
"""

PREFIX = """
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX ntu:  <http://ntu.edu.vn/feedback_system#>
"""

SPARQL_V8 = {

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_01_CountBySeverity": {
    "title": "Phân bố số lượng phản hồi theo mức độ nghiêm trọng",
    "description": "Đếm tổng số phản hồi theo từng SeverityLevel (Low/Medium/High/Critical).",
    "new_elements": ["SeverityLevel", "hasSeverityLevel"],
    "query": PREFIX + """
SELECT ?sevLabel (COUNT(?fb) AS ?soLuong)
WHERE {
    ?fb  rdf:type        ntu:Feedback ;
         ntu:hasSeverityLevel ?sev .
    ?sev rdfs:label      ?sevLabel .
}
GROUP BY ?sevLabel
ORDER BY DESC(?soLuong)
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_02_CriticalFeedbacks": {
    "title": "Danh sách phản hồi nghiêm trọng (Critical) cần xử lý ngay",
    "description": "Truy xuất nội dung, cảm xúc và giảng viên liên quan của tất cả "
                   "phản hồi Critical — đầu vào ưu tiên cho ban quản lý.",
    "new_elements": ["SeverityLevel", "hasSeverityLevel"],
    "query": PREFIX + """
SELECT ?fb ?content ?emotion ?gvCode ?hpCode
WHERE {
    ?fb  rdf:type             ntu:Feedback ;
         ntu:hasContent       ?content ;
         ntu:hasSeverityLevel ntu:Sev_Critical .
    OPTIONAL { ?fb ntu:hasEmotion    ?emo . ?emo rdfs:label ?emotion . }
    OPTIONAL { ?fb ntu:aboutLecturer ?gv  . ?gv  ntu:hasLecturerCode ?gvCode . }
    OPTIONAL { ?fb ntu:aboutCourse   ?hp  . ?hp  ntu:hasCourseCode   ?hpCode . }
}
ORDER BY ?gvCode
LIMIT 50
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_03_HighCriticalByLecturer": {
    "title": "Số phản hồi High/Critical theo từng giảng viên",
    "description": "Xếp hạng giảng viên theo số phản hồi nghiêm trọng — "
                   "hỗ trợ ưu tiên hỗ trợ chuyên môn.",
    "new_elements": ["SeverityLevel", "hasSeverityLevel"],
    "query": PREFIX + """
SELECT ?gvCode (COUNT(?fb) AS ?soFBNghiemTrong)
WHERE {
    ?fb  rdf:type             ntu:Feedback ;
         ntu:hasSeverityLevel ?sev ;
         ntu:aboutLecturer    ?gv .
    ?gv  ntu:hasLecturerCode  ?gvCode .
    FILTER(?sev IN (ntu:Sev_High, ntu:Sev_Critical))
}
GROUP BY ?gvCode
ORDER BY DESC(?soFBNghiemTrong)
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_04_SeverityBySentiment": {
    "title": "Ma trận Sentiment × SeverityLevel",
    "description": "Kiểm tra tương quan giữa nhãn Sentiment và SeverityLevel — "
                   "đánh giá độ nhất quán của pipeline gán nhãn tự động.",
    "new_elements": ["SeverityLevel", "hasSeverityLevel"],
    "query": PREFIX + """
SELECT ?sentLabel ?sevLabel (COUNT(?fb) AS ?count)
WHERE {
    ?fb   rdf:type             ntu:Feedback ;
          ntu:hasSentiment     ?sent ;
          ntu:hasSeverityLevel ?sev .
    ?sent rdfs:label           ?sentLabel .
    ?sev  rdfs:label           ?sevLabel .
}
GROUP BY ?sentLabel ?sevLabel
ORDER BY ?sentLabel ?sevLabel
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_05_RecommendationsByPriority": {
    "title": "Khuyến nghị theo thứ tự ưu tiên (priority score giảm dần)",
    "description": "Liệt kê toàn bộ Recommendation còn Open/InProgress, "
                   "sắp xếp theo hasPriorityScore — giúp ban lãnh đạo phân bổ nguồn lực.",
    "new_elements": ["Recommendation", "hasPriorityScore", "hasResolutionStatus"],
    "query": PREFIX + """
SELECT ?rec ?actionText ?priority ?status ?topicLabel
WHERE {
    ?rec rdf:type                ntu:Recommendation ;
         ntu:hasActionText       ?actionText ;
         ntu:hasPriorityScore    ?priority ;
         ntu:hasResolutionStatus ?status .
    FILTER(?status IN ("Open", "InProgress"))
    OPTIONAL {
        ?fb ntu:recommendsAction ?rec ;
            ntu:aboutTopic       ?topic .
        ?topic rdfs:label        ?topicLabel .
    }
}
ORDER BY DESC(?priority)
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_06_FeedbacksLinkedToRec": {
    "title": "Phản hồi tiêu cực được liên kết với khuyến nghị cụ thể",
    "description": "Với mỗi Recommendation, đếm số phản hồi Negative đang trỏ vào nó — "
                   "đo mức độ cấp thiết thực tế của từng khuyến nghị.",
    "new_elements": ["Recommendation", "recommendsAction", "hasPriorityScore"],
    "query": PREFIX + """
SELECT ?actionText ?priority ?status (COUNT(?fb) AS ?soFBNegative)
WHERE {
    ?rec rdf:type                ntu:Recommendation ;
         ntu:hasActionText       ?actionText ;
         ntu:hasPriorityScore    ?priority ;
         ntu:hasResolutionStatus ?status .
    ?fb  ntu:recommendsAction   ?rec ;
         ntu:hasSentiment        ntu:Sent_Negative .
}
GROUP BY ?actionText ?priority ?status
ORDER BY DESC(?soFBNegative)
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_07_ResolvedVsOpen": {
    "title": "Tỷ lệ Recommendation theo trạng thái xử lý",
    "description": "Tổng quan tiến độ: bao nhiêu khuyến nghị đã Resolved/Closed vs còn Open.",
    "new_elements": ["Recommendation", "hasResolutionStatus"],
    "query": PREFIX + """
SELECT ?status (COUNT(?rec) AS ?soLuong)
WHERE {
    ?rec rdf:type                ntu:Recommendation ;
         ntu:hasResolutionStatus ?status .
}
GROUP BY ?status
ORDER BY ?status
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_08_NegativeFeedbackByDepartment": {
    "title": "Phản hồi tiêu cực theo Bộ môn/Khoa",
    "description": "Phân tích qua chuỗi: Feedback → Course → Department → Faculty. "
                   "Xác định khoa/bộ môn nhận nhiều phản hồi tiêu cực nhất.",
    "new_elements": ["Department", "Faculty", "belongsToDepartment", "belongsToFaculty"],
    "query": PREFIX + """
SELECT ?deptLabel ?facLabel (COUNT(?fb) AS ?soFBTieuCuc)
WHERE {
    ?fb   rdf:type              ntu:Feedback ;
          ntu:hasSentiment      ntu:Sent_Negative ;
          ntu:aboutCourse       ?hp .
    ?hp   ntu:belongsToDepartment ?dept .
    ?dept rdfs:label            ?deptLabel ;
          ntu:belongsToFaculty  ?fac .
    ?fac  rdfs:label            ?facLabel .
}
GROUP BY ?deptLabel ?facLabel
ORDER BY DESC(?soFBTieuCuc)
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_09_TopicsByFaculty": {
    "title": "Phân bố chủ đề phản hồi theo Khoa (Faculty)",
    "description": "Mỗi khoa nhận phản hồi nào nhiều nhất — "
                   "giúp định hướng cải tiến đặc thù cho từng khoa.",
    "new_elements": ["Department", "Faculty", "belongsToDepartment", "belongsToFaculty"],
    "query": PREFIX + """
SELECT ?facLabel ?topicLabel (COUNT(?fb) AS ?count)
WHERE {
    ?fb    rdf:type               ntu:Feedback ;
           ntu:aboutTopic         ?topic ;
           ntu:aboutCourse        ?hp .
    ?hp    ntu:belongsToDepartment ?dept .
    ?dept  ntu:belongsToFaculty   ?fac .
    ?fac   rdfs:label             ?facLabel .
    ?topic rdfs:label             ?topicLabel .
}
GROUP BY ?facLabel ?topicLabel
ORDER BY ?facLabel DESC(?count)
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_10_TrendByAcademicTerm": {
    "title": "Xu hướng Sentiment theo từng kỳ học",
    "description": "So sánh tỷ lệ Positive/Neutral/Negative qua các kỳ học — "
                   "theo dõi tiến bộ cải thiện chất lượng giảng dạy.",
    "new_elements": ["AcademicTerm", "belongsToAcademicTerm", "hasAcademicYear"],
    "query": PREFIX + """
SELECT ?academicYear ?semesterNum ?sentLabel (COUNT(?fb) AS ?count)
WHERE {
    ?fb   rdf:type                 ntu:Feedback ;
          ntu:hasSentiment         ?sent ;
          ntu:belongsToAcademicTerm ?term .
    ?sent rdfs:label               ?sentLabel .
    ?term ntu:hasAcademicYear      ?academicYear ;
          ntu:hasSemester          ?semesterNum .
}
GROUP BY ?academicYear ?semesterNum ?sentLabel
ORDER BY ?academicYear ?semesterNum ?sentLabel
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_11_FeedbackBySourceAndOrigin": {
    "title": "Phân bố phản hồi theo kênh thu thập và xuất xứ dữ liệu",
    "description": "Kiểm tra tỷ trọng dữ liệu theo FeedbackSource và DataOrigin — "
                   "hỗ trợ đánh giá bias trong tập dữ liệu.",
    "new_elements": ["FeedbackSource", "DataOrigin", "generatedFromSource",
                     "hasDataOrigin", "hasSourceName", "hasOriginLabel"],
    "query": PREFIX + """
SELECT ?sourceName ?originLabel (COUNT(?fb) AS ?count)
WHERE {
    ?fb     rdf:type              ntu:Feedback ;
            ntu:generatedFromSource ?src ;
            ntu:hasDataOrigin     ?orig .
    ?src    ntu:hasSourceName     ?sourceName .
    ?orig   ntu:hasOriginLabel    ?originLabel .
}
GROUP BY ?sourceName ?originLabel
ORDER BY ?sourceName
"""},

# ─────────────────────────────────────────────────────────────────────────────
"Q_V8_12_DashboardSummary": {
    "title": "Dashboard tổng hợp đa chiều — V8 Full",
    "description": "Query tổng hợp cho dashboard quản lý: kết nối Feedback với "
                   "Severity + Recommendation + Department + AcademicTerm + Source.",
    "new_elements": ["tất cả thành phần V8"],
    "query": PREFIX + """
SELECT
    ?gvCode ?hpCode ?deptLabel
    ?academicYear ?semester
    ?sentLabel ?sevLabel ?topicLabel
    ?sourceName ?priority ?recStatus
WHERE {
    ?fb  rdf:type                  ntu:Feedback ;
         ntu:hasSentiment          ?sent ;
         ntu:hasSeverityLevel      ?sev ;
         ntu:belongsToAcademicTerm ?term ;
         ntu:generatedFromSource   ?src .

    ?sent  rdfs:label              ?sentLabel .
    ?sev   rdfs:label              ?sevLabel .
    ?term  ntu:hasAcademicYear     ?academicYear ;
           ntu:hasSemester         ?semester .
    ?src   ntu:hasSourceName       ?sourceName .

    OPTIONAL { ?fb ntu:aboutLecturer    ?gv   . ?gv   ntu:hasLecturerCode   ?gvCode . }
    OPTIONAL {
        ?fb ntu:aboutCourse          ?hp   .
        ?hp ntu:hasCourseCode        ?hpCode ;
            ntu:belongsToDepartment  ?dept .
        ?dept rdfs:label             ?deptLabel .
    }
    OPTIONAL { ?fb ntu:aboutTopic        ?topic . ?topic rdfs:label  ?topicLabel . }
    OPTIONAL {
        ?fb  ntu:recommendsAction    ?rec .
        ?rec ntu:hasPriorityScore    ?priority ;
             ntu:hasResolutionStatus ?recStatus .
    }

    FILTER(?sev IN (ntu:Sev_High, ntu:Sev_Critical))
}
ORDER BY DESC(?priority) ?academicYear ?gvCode
LIMIT 100
"""},
}

# ─── XUẤT FILE TXT ────────────────────────────────────────────────────────────
import os
out_path = os.path.join(os.path.dirname(__file__), "SPARQL_Queries_V8.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("# SPARQL QUERIES V8 — NTU Feedback System Ontology Mở Rộng\n")
    f.write("# " + "=" * 72 + "\n\n")
    for qid, qdata in SPARQL_V8.items():
        f.write(f"## {qid}\n")
        f.write(f"# Tiêu đề   : {qdata['title']}\n")
        f.write(f"# Mô tả     : {qdata['description']}\n")
        f.write(f"# Thành phần mới: {', '.join(qdata['new_elements'])}\n")
        f.write(qdata["query"])
        f.write("\n" + "# " + "-" * 72 + "\n\n")

print(f"✓ Đã ghi {len(SPARQL_V8)} queries → {out_path}")
print("\nDanh sách queries:")
for qid, qdata in SPARQL_V8.items():
    print(f"  {qid:40s} | {qdata['title']}")
