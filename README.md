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
