from typing import List

from pydantic import BaseModel, field_validator

from agents.llm_client import LLMClient
from agents.utils import coerce_string_list


class SummaryResult(BaseModel):
    summary: str
    action_items: List[str]

    @field_validator("action_items", mode="before")
    @classmethod
    def _coerce_action_items(cls, v):
        return coerce_string_list(v)


SYSTEM_PROMPT = """Ты суммаризируешь звонок в банковский контакт-центр по транскрипту.
summary: краткое резюме звонка, 3-5 предложений.
action_items: список конкретных дальнейших действий после звонка (может быть пустым).

Ответь СТРОГО в формате JSON, без пояснений и markdown:
{"summary": "...", "action_items": ["..."]}"""


def summarize(transcript_text: str, llm: LLMClient) -> SummaryResult:
    return llm.call_structured(SYSTEM_PROMPT, transcript_text, SummaryResult, "summarizer")
