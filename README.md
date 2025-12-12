# 🌱 FarmPest Vision — Offline Pest Detection & IPM Assistant

FarmPest Vision là một hệ thống AI chạy **hoàn toàn offline**, kết hợp giữa:

- 🐛 **Nhận diện sâu bệnh bằng CNN (TensorFlow/Keras)**
- 🤖 **Tư vấn IPM & xử lý hữu cơ bằng LLM chạy local (LM Studio + llama-3.2-1b-instruct)**
- 🖥 **Giao diện trực quan với Streamlit**

Hệ thống phù hợp triển khai ngoài nông trại, nơi **không có Internet**, phần cứng yếu và cần tốc độ nhanh, chính xác, chi phí thấp.

---

# 🚀 Tính năng chính

### 🐛 1. Nhận diện sâu bệnh bằng mô hình CNN
- Input: ảnh JPG/PNG
- Output:
  - Tên sâu bệnh
  - Độ tin cậy
  - Chỉ số class
- Mô hình: `pest_classification_model.keras`

---

### 🤖 2. Tư vấn IPM bằng LLM offline
- Chạy qua **LM Studio API (OpenAI-compatible)**  
- Model đề xuất: **llama-3.2-1b-instruct** (nhẹ, chạy tốt trên máy yếu)
- LLM có khả năng:
  - Hiểu câu hỏi từ nông dân
  - Phân tích yêu cầu
  - Kích hoạt **function calling** khi cần phân tích ảnh
  - Tư vấn IPM, xử lý hữu cơ, hướng dẫn canh tác an toàn

---

### 🧠 3. Function Calling offline
Luồng xử lý giống OpenAI Function Calling, nhưng chạy **100% offline**:

# 🧱 Kiến trúc hệ thống

```svg
<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="650" viewBox="0 0 1200 650" xmlns="http://www.w3.org/2000/svg">

<rect x="0" y="0" width="1200" height="650" fill="#f7f7f7"/>

<defs>
<style type="text/css"><![CDATA[
.block { fill:#ffffff;stroke:#333;stroke-width:2;rx:12;ry:12; }
.decision { fill:#fdf4c5;stroke:#333;stroke-width:2; }
.label { font-family:Arial;font-size:14px;fill:#222;text-anchor:middle; }
.title { font-weight:bold; }
.small { font-size:12px;fill:#444; }
.arrow { stroke:#555;stroke-width:2;marker-end:url(#arrowhead);fill:none; }
]]></style>
<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
<polygon points="0 0, 10 3.5, 0 7" fill="#555"/>
</marker>
</defs>

<!-- User -->
<rect class="block" x="70" y="60" width="200" height="80" />
<text class="label title" x="170" y="90">Người dùng</text>
<text class="label small" x="170" y="112">(Upload ảnh + câu hỏi)</text>

<!-- UI -->
<rect class="block" x="350" y="60" width="220" height="90"/>
<text class="label title" x="460" y="90">UI (Streamlit)</text>
<text class="label small" x="460" y="112">Hiển thị ảnh & gửi yêu cầu</text>

<!-- LLM -->
<rect class="block" x="660" y="60" width="240" height="90"/>
<text class="label title" x="780" y="90">LLM (LM Studio)</text>
<text class="label small" x="780" y="112">llama-3.2-1b-instruct</text>

<!-- Decision -->
<polygon class="decision" points="780,210 880,260 780,310 680,260"/>
<text class="label title" x="780" y="248">Quyết định</text>
<text class="label small" x="780" y="268">Có cần gọi CNN?</text>

<!-- Function call -->
<rect class="block" x="720" y="340" width="220" height="90"/>
<text class="label title" x="830" y="370">Function Calling</text>
<text class="label small" x="830" y="392">classify_pest()</text>

<!-- CNN -->
<rect class="block" x="720" y="470" width="220" height="90"/>
<text class="label title" x="830" y="500">CNN Model</text>
<text class="label small" x="830" y="522">pest_classification_model.keras</text>

<!-- Result -->
<rect class="block" x="420" y="470" width="230" height="90"/>
<text class="label title" x="535" y="500">Kết quả CNN</text>

<!-- LLM synthesis -->
<rect class="block" x="350" y="330" width="260" height="100"/>
<text class="label title" x="480" y="360">LLM tổng hợp</text>
<text class="label small" x="480" y="382">Giải thích sâu + IPM</text>

<!-- Output -->
<rect class="block" x="70" y="330" width="220" height="90"/>
<text class="label title" x="180" y="360">Trả lời</text>

<!-- Arrows -->
<line class="arrow" x1="270" y1="100" x2="350" y2="100"/>
<line class="arrow" x1="570" y1="100" x2="660" y2="100"/>
<line class="arrow" x1="780" y1="150" x2="780" y2="210"/>
<line class="arrow" x1="830" y1="310" x2="830" y2="340"/>
<line class="arrow" x1="730" y1="310" x2="610" y2="330"/>
<line class="arrow" x1="830" y1="430" x2="830" y2="470"/>
<line class="arrow" x1="720" y1="515" x2="650" y2="515"/>
<line class="arrow" x1="535" y1="470" x2="535" y2="430"/>
<line class="arrow" x1="350" y1="380" x2="290" y2="380"/>

</svg>

FarmPest-Vision/
│
├── app.py                      # Giao diện Streamlit
├── pest_model.py               # Load CNN + predict_pest()
├── fc_agent.py                 # Function calling offline
├── llm_client.py               # Kết nối LM Studio API
├── pest_classification_model.keras
│
├── README.md
└── requirements.txt

👨‍💻 Tác giả
Huynh Tan Loc
Vietnam
