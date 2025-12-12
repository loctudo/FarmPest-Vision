# 🌱 FarmPest Vision — Offline Pest Detection

FarmPest Vision is a fully offline AI system designed for real-world agricultural environments, where internet connectivity is limited or unavailable. The system integrates:

- 🐛 Pest detection using a CNN model (TensorFlow/Keras)
- 🤖 IPM (Integrated Pest Management) recommendations powered by a local LLM (LM Studio + llama-3.2-1b-instruct)
- 🖥 A simple and interactive Streamlit interface

It is optimized for low-compute devices and field deployment, providing fast, reliable, and low-cost pest diagnosis and decision support.
---

# 🚀 Key Features

### 🐛 1. Pest Classification via CNN
- Input: JPG/PNG images
- Output:
  - Predicted pest species
  - Confidence score
  - Class index
- Model used: pest_classification_model.keras

---

### 🤖 2. Offline Recommendation (Local LLM)
- Runs through LM Studio's local API (OpenAI-compatible) 
- Recommended model: llama-3.2-1b-instruct (lightweight, fast, suitable for offline/edge devices)
- The LLM can
  - Understand users' natural-language questions
  - Analyze context and intent
  - Trigger function calling when image analysis is needed
  - Provide actionable IPM guidance, organic treatments, and safety recommendations

---

### 🧠 3. Offline Function Calling
FarmPest Vision implements an OpenAI-style function-calling pipeline, but runs entirely offline


👨‍💻 Author
Huynh Tan Loc
Vietnam
