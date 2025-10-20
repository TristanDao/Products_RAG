from typing import List, Dict

class Reflection:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def rewrite(self, messages: List[Dict], current_query: str) ->str:
        chat_history = [mes for mes in messages if mes['role'] in ('user', 'assistant')][-10:]
        history_text = ""
        for mes in chat_history:
            role = 'Người dùng' if mes['role'] == "user" else "Trợ lý"
            history_text += f"{role}: {mes['content']}\n"
        history_text += f"Khách: {current_query}\n"

        prompt = [
            {
                "role": "system",
                "content": (
                    "Dưới đây là lịch sử hội thoại và câu hỏi mới nhất của người dùng."
                    "Hãy viết lại câu hỏi sao cho nó trở thành một câu hỏi độc lập, "
                    "có thể hiểu được mà không cần tham chiếu đến ngữ cảnh trước đó. "
                    "Không trả lời câu hỏi. Chỉ trả về cau hỏi được viết lại."
                )
            },
            {"role": "user", "content": history_text}
        ]
        response = self.llm_client.chat.completions.create(
            model ="gpt-4o-mini",
            messages = prompt,
            temperature = 0
        )
        rewrite = response.choices[0].message.content.strip()
        print(f"Câu hỏi được viết lại: {rewrite}")
        return rewrite