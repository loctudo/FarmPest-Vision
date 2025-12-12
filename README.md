flowchart TD

    %% ==== STYLES (giống draw.io) ====
    classDef block fill:#ffffff,stroke:#333,stroke-width:2px,rx:8px,ry:8px,font-size:16px;
    classDef decision fill:#fdf4c5,stroke:#333,stroke-width:2px,rx:8px,ry:8px,font-size:16px;

    %% ==== NODES ====
    User[Người dùng<br/>(Upload ảnh + đặt câu hỏi)]:::block
    UI[UI (Streamlit)<br/>• Hiển thị ảnh<br/>• Nút chạy CNN<br/>• Gửi câu hỏi]:::block
    LLM[LLM (LM Studio)<br/>llama-3.2-1b-instruct]:::block
    Decide{LLM phân tích yêu cầu<br/>Có cần phân tích ảnh không?}:::decision
    CallCNN[Function Calling<br/>classify_pest()]:::block
    CNN[CNN Model<br/>pest_classification_model.keras]:::block
    Result[Kết quả CNN<br/>• Tên sâu bệnh<br/>• Độ tin cậy]:::block
    FinalLLM[LLM phân tích + tổng hợp:<br/>• Giải thích sâu bệnh<br/>• Gợi ý IPM<br/>• Xử lý hữu cơ]:::block
    Output[Trả lời người dùng<br/>(UI Chat Output)]:::block

    %% ==== EDGES ====
    User --> UI
    UI --> LLM

    LLM --> Decide

    Decide -->|Có| CallCNN
    CallCNN --> CNN
    CNN --> Result
    Result --> LLM

    Decide -->|Không| LLM

    LLM --> FinalLLM
    FinalLLM --> Output
