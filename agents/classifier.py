from pydantic import BaseModel

from agents.llm_client import LLMClient


class ClassificationResult(BaseModel):
    topic: str
    priority: str


SYSTEM_PROMPT = """Ты — классификатор обращений в банковский контакт-центр.
По транскрипту звонка определи:
- topic: одна из тем — "кредиты", "карты", "переводы", "жалобы", "другое"
- priority: "low", "medium" или "high" (жалобы и упоминания финансовых потерь клиента — высокий приоритет)

Ответь СТРОГО в формате JSON, без пояснений и markdown:
{"topic": "...", "priority": "..."}"""


def classify(transcript_text: str, llm: LLMClient) -> ClassificationResult:
    return llm.call_structured(SYSTEM_PROMPT, transcript_text, ClassificationResult, "classifier")
