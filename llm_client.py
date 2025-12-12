import os
import json
import requests
from typing import List, Dict, Any

# ===============================================================
# CẤU HÌNH CHUẨN CHO LM STUDIO (THEO ẢNH BẠN GỬI)
# ===============================================================

LM_STUDIO_URL = os.getenv(
    "LM_STUDIO_URL",
    "http://127.0.0.1:1234/v1/chat/completions"
)

MODEL_NAME = os.getenv(
    "LM_STUDIO_MODEL",
    "llama-3.2-1b-instruct"
)

LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

# ===============================================================
# HÀM GỌI CHAT COMPLETION
# ===============================================================

def chat_with_llama(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LM_STUDIO_API_KEY}",
    }

    response = requests.post(
        LM_STUDIO_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]
