import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ==========================
# 1. LOAD MODEL
# ==========================
def load_cnn_model(model_path: str = "pest_classification_model.keras"):
    model = load_model(model_path)
    return model


# ==========================
# 2. DANH SÁCH CLASS NAMES
# ==========================

CLASS_NAMES = [
    "ants",
    "bees",
    "beetle",
    "catterpillar",
    "earthworms",
    "earwig",
    "grasshopper",
    "moth",
    "slug",
    "snail",
    "wasp",
    "weevil"
]

# ==========================
# 3. HÀM TIỀN XỬ LÝ ẢNH
# ==========================

IMG_HEIGHT = 300
IMG_WIDTH = 225

def preprocess_image(pil_image: Image.Image) -> tf.Tensor:
    img = np.array(pil_image, dtype=np.float32)
    img_resized = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img_batch = tf.expand_dims(img_resized, axis=0)
    return img_batch


# ==========================
# 4. HÀM DỰ ĐOÁN
# ==========================

def predict_pest(model, pil_image: Image.Image):

    input_tensor = preprocess_image(pil_image)

    # Dự đoán
    preds = model.predict(input_tensor)
    preds = preds[0]

    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    if 0 <= class_index < len(CLASS_NAMES):
        pest_name = CLASS_NAMES[class_index]
    else:
        pest_name = f"Unknown-{class_index}"

    return {
        "pest_name": pest_name,
        "confidence": confidence,
        "class_index": class_index
    }
