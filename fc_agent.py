import json
from typing import Dict, Any, Optional

from PIL import Image
from pest_model import predict_pest
from llm_client import chat_with_llama


# ==========================
# 1. Bước chọn tool (tool selection)
# ==========================

def select_tool_via_llm(user_message: str, has_image: bool) -> Dict[str, Any]:
    """
    Gửi yêu cầu tới LLM để quyết định có gọi tool classify_pest hay không.

    LLM phải trả về JSON với format:
        {
          "tool": "classify_pest" | null,
          "arguments": {}
        }

    Trong context, ta cho LLM biết biến HAS_IMAGE để nó biết có ảnh hay không.
    """

    system_prompt = (
        "You are a tool selector for a pest management assistant.\n"
        "There is exactly one tool available:\n"
        "  - classify_pest(): uses the CURRENT uploaded image (if any) to "
        "    classify the pest with a CNN model.\n\n"
        "You MUST decide whether to call this tool for the user's request.\n"
        "Rules:\n"
        "  - If HAS_IMAGE is false, you cannot call classify_pest.\n"
        "  - If HAS_IMAGE is true and the user asks about the image, pest, "
        "    'what pest is this', 'classify', etc., you SHOULD call classify_pest.\n"
        "  - If the user only asks conceptual questions (no need to inspect "
        "    the current image), do not call any tool.\n\n"
        "Respond ONLY with valid JSON in this exact schema:\n"
        "{\n"
        '  \"tool\": \"classify_pest\" or null,\n'
        "  \"arguments\": { }\n"
        "}\n"
    )

    user_prompt = (
        f"HAS_IMAGE={str(has_image).lower()}.\n\n"
        f"User message:\n{user_message}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw = chat_with_llama(messages, temperature=0.0, max_tokens=256)

    try:
        parsed = json.loads(raw)
        tool = parsed.get("tool", None)
        arguments = parsed.get("arguments", {})
        if tool not in ("classify_pest", None):
            tool = None
        if not isinstance(arguments, dict):
            arguments = {}
        return {"tool": tool, "arguments": arguments}
    except Exception:
        return {"tool": None, "arguments": {}}


# ==========================
# 2. Bước sinh câu trả lời cuối cùng
# ==========================

def generate_final_answer_via_llm(
    user_message: str,
    farm_context: Dict[str, Any],
    pest_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Dùng LLM để sinh câu trả lời cuối cùng cho người dùng,
    dựa trên:
      - câu hỏi ban đầu,
      - bối cảnh trang trại,
      - kết quả CNN (pest_result), nếu có.
    """

    farm_name = farm_context.get("farm_name", "Organic Farm")
    crop_type = farm_context.get("crop_type", "cây trồng")
    location = farm_context.get("location", "")

    system_prompt = (
        "You are an AI assistant for organic pest management on farms.\n"
        "Your tasks:\n"
        "1) Interpret the CNN classification result (if provided).\n"
        "2) Explain to the farmer what the pest is and how certain the model is.\n"
        "3) Suggest practical IPM and organic control strategies for the given crop.\n"
        "4) Answer in Vietnamese, using clear and concise language.\n"
        "5) If no pest_result is provided, still try to give general advice based on the question.\n"
    )

    context_lines = [
        f"Tên trang trại: {farm_name}",
        f"Cây trồng chính: {crop_type}",
    ]
    if location:
        context_lines.append(f"Khu vực: {location}")

    context_text = "\n".join(context_lines)

    if pest_result is not None:
        pest_json = json.dumps(pest_result, ensure_ascii=False, indent=2)
        assistant_context = (
            f"Thông tin bối cảnh trang trại:\n{context_text}\n\n"
            f"Kết quả phân loại từ mô hình CNN (pest_result):\n{pest_json}\n"
        )
    else:
        assistant_context = (
            f"Thông tin bối cảnh trang trại:\n{context_text}\n\n"
            "Hiện chưa có kết quả phân loại pest_result từ mô hình CNN.\n"
        )

    user_prompt = (
        f"{assistant_context}\n"
        "Câu hỏi của người nông dân:\n"
        f"{user_message}\n\n"
        "Hãy trả lời bằng tiếng Việt, bao gồm:\n"
        "- Giải thích (nếu có pest_result) sâu bệnh là gì, mức độ chắc chắn.\n"
        "- Các bước IPM (khảo sát, giảm mật độ, dùng thiên địch, quản lý canh tác,...).\n"
        "- Gợi ý biện pháp xử lý hữu cơ cụ thể nhưng an toàn.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = chat_with_llama(messages, temperature=0.3, max_tokens=768)
    return answer


# ==========================
# 3. Hàm tổng – cho UI gọi
# ==========================

def run_pest_fc(
    user_message: str,
    farm_context: Dict[str, Any],
    pil_image: Optional[Image.Image],
    cnn_model,
) -> str:
    """
    Hàm tổng hợp cho UI:

    - Nhận câu hỏi từ người dùng (user_message).
    - Biết bối cảnh trang trại (farm_context).
    - Biết ảnh hiện tại (pil_image) nếu có, và model CNN đã load (cnn_model).
    - Gọi LLM để quyết định có dùng tool classify_pest hay không.
    - Nếu cần, gọi predict_pest(cnn_model, pil_image).
    - Sau đó gọi LLM lần 2 để sinh câu trả lời cuối cùng bằng tiếng Việt.

    Trả về: câu trả lời dạng text để hiển thị trên UI.
    """

    has_image = pil_image is not None
    tool_call = select_tool_via_llm(user_message, has_image=has_image)

    pest_result: Optional[Dict[str, Any]] = None

    if tool_call["tool"] == "classify_pest":
        if not has_image or cnn_model is None:
            return (
                "Bạn yêu cầu phân tích từ hình ảnh, nhưng hệ thống hiện không có ảnh "
                "hoặc mô hình CNN chưa được tải. Hãy kiểm tra lại việc upload ảnh và "
                "khởi tạo mô hình. "
                "Tuy nhiên, bạn vẫn có thể mô tả triệu chứng để tôi tư vấn chung.\n\n"
                + generate_final_answer_via_llm(
                    user_message=user_message,
                    farm_context=farm_context,
                    pest_result=None,
                )
            )

        # Gọi tool classify_pest (CNN)
        pest_result = predict_pest(cnn_model, pil_image)

    final_answer = generate_final_answer_via_llm(
        user_message=user_message,
        farm_context=farm_context,
        pest_result=pest_result,
    )

    return final_answer
