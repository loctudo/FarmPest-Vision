import io
import streamlit as st
from PIL import Image

from pest_model import load_cnn_model, predict_pest
from fc_agent import run_pest_fc


# ==========================
# CẤU HÌNH
# ==========================
st.set_page_config(
    page_title="FarmPest Vision | Pest Detection + LLM",
    page_icon="🪲",
    layout="wide"
)

# SESSION STATE
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "cnn_result" not in st.session_state:
    st.session_state.cnn_result = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "farm_info" not in st.session_state:
    st.session_state.farm_info = {
        "farm_name": "Organic Farm A",
        "crop_type": "Tomato",
        "location": "Vietnam",
    }


# ==========================
# TẢI MODEL CNN
# ==========================
@st.cache_resource
def load_model_cached():
    return load_cnn_model("pest_classification_model.keras")

cnn_model = load_model_cached()


# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.title("📷 Ảnh sâu bệnh")

    uploaded_img = st.file_uploader("Chọn ảnh JPG/PNG", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img_bytes = uploaded_img.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.session_state.uploaded_image = pil
        st.success("Ảnh đã được tải lên!")

    st.markdown("---")
    st.subheader("🌱 Thông tin trang trại")

    farm = st.text_input("Tên trang trại", st.session_state.farm_info["farm_name"])
    crop = st.text_input("Cây trồng chính", st.session_state.farm_info["crop_type"])
    loc = st.text_input("Khu vực", st.session_state.farm_info["location"])

    st.session_state.farm_info.update({
        "farm_name": farm,
        "crop_type": crop,
        "location": loc
    })


# ==========================
# MAIN UI (2 CỘT)
# ==========================
st.title("🪲 FarmPest Vision — Phân loại côn trùng bằng AI")

left, right = st.columns([1.2, 1])

# ===== LEFT COLUMN =====
with left:
    st.subheader("Ảnh đầu vào")

    if st.session_state.uploaded_image:
        st.image(
            st.session_state.uploaded_image,
            caption="Ảnh sâu bệnh",
            width=450
        )
    else:
        st.info("Hãy upload ảnh sâu bệnh từ sidebar.")




# ===== RIGHT COLUMN =====
with right:
    # Button detect
    if st.session_state.uploaded_image:
        if st.button("🔍 Phân tích ảnh bằng CNN"):
            with st.spinner("Đang dự đoán..."):
                st.session_state.cnn_result = predict_pest(
                    cnn_model, st.session_state.uploaded_image
                )

    st.markdown("### Kết quả dự đoán (CNN)")
    if st.session_state.cnn_result:
        r = st.session_state.cnn_result
        st.write(f"**Loài:** `{r['pest_name']}`")
        st.write(f"**Độ tin cậy:** `{r['confidence']:.4f}`")
        st.write(f"**Class index:** `{r['class_index']}`")
    else:
        st.caption("Chưa có kết quả.")


# ==========================
# KHUNG LLM
# ==========================
st.markdown("---")
st.subheader("🤖 Trợ lý LLM — Hỏi đáp về sâu bệnh")

chat_box = st.container()

with chat_box:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar")):
            st.write(msg["content"])

# INPUT CHAT BOTTOM
user_text = st.chat_input("Đặt câu hỏi…")

if user_text:
    # Display user message
    st.session_state.chat_messages.append(
        {"role": "user", "content": user_text, "avatar": "🧑‍🌾"}
    )

    with st.spinner("LLM đang trả lời…"):
        reply = run_pest_fc(
            user_message=user_text,
            farm_context=st.session_state.farm_info,
            pil_image=st.session_state.uploaded_image,
            cnn_model=cnn_model,
        )

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": reply, "avatar": "🤖"}
    )

    st.rerun()
