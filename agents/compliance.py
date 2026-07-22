from typing import List

from pydantic import BaseModel, field_validator

from agents.llm_client import LLMClient
from agents.utils import coerce_string_list


class ComplianceResult(BaseModel):
    passed: bool
    issues: List[str]

    @field_validator("issues", mode="before")
    @classmethod
    def _coerce_issues(cls, v):
        return coerce_string_list(v)


SYSTEM_PROMPT = """Ты проверяешь звонок банковского оператора на соответствие compliance-требованиям.
Ищи:
- запрещённые или некорректные обещания (например, гарантии одобрения кредита)
- отсутствие обязательных disclaimers там, где они нужны (условия кредита/вклада, ставки, комиссии)
- вводящие в заблуждение формулировки при предложении продукта

passed: true, если нарушений не найдено.
issues: список конкретных найденных проблем (пустой список, если всё чисто).

Ответь СТРОГО в формате JSON, без пояснений и markdown:
{"passed": true, "issues": []}"""


def check_compliance(transcript_text: str, llm: LLMClient) -> ComplianceResult:
    return llm.call_structured(SYSTEM_PROMPT, transcript_text, ComplianceResult, "compliance")
