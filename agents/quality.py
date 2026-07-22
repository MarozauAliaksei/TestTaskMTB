from pydantic import BaseModel

from agents.llm_client import LLMClient


class QualityChecklist(BaseModel):
    greeting: bool
    need_detection: bool
    solution_provided: bool
    farewell: bool


class QualityResult(BaseModel):
    total: int
    checklist: QualityChecklist


SYSTEM_PROMPT = """Ты оцениваешь качество работы оператора банковского контакт-центра по транскрипту звонка.
Проверь чек-лист (true/false):
- greeting: оператор поздоровался и представился
- need_detection: оператор выяснил, что нужно клиенту
- solution_provided: клиенту предложено решение или дан ответ на вопрос
- farewell: оператор попрощался

total — общая оценка от 0 до 100 (учитывай не только чек-лист, но и тон и вежливость).

Ответь СТРОГО в формате JSON, без пояснений и markdown:
{"total": 0, "checklist": {"greeting": true, "need_detection": true, "solution_provided": true, "farewell": true}}"""


def evaluate_quality(transcript_text: str, llm: LLMClient) -> QualityResult:
    return llm.call_structured(SYSTEM_PROMPT, transcript_text, QualityResult, "quality")
