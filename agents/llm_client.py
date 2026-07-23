import json
import logging
import re
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger("agents.llm_client")

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Неразрывный пробел, zero-width space/joiner, BOM — частые "невидимки" при
# копипасте API-ключей/имён моделей из браузера.
_INVISIBLE_CHARS_RE = re.compile("[\u00a0\u200b\u200c\u200d\ufeff]")


def _strip_invisible_chars(value: str) -> str:
    return _INVISIBLE_CHARS_RE.sub("", value).strip()


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "ollama"):
        # Ключ и имя модели иногда попадают сюда через copy-paste из браузера
        # (например, из консоли Groq) и могут содержать невидимые символы —
        # неразрывный пробел, zero-width space и т.п. HTTP-заголовки (в
        # частности Authorization) обязаны быть чистым ASCII, поэтому такой
        # символ ломает КАЖДЫЙ запрос с невнятной ошибкой кодировки. Чистим
        # заранее, а не дожидаемся крипто-ошибки в рантайме.
        api_key = _strip_invisible_chars(api_key)
        model = _strip_invisible_chars(model)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def call_structured(self, system_prompt: str, user_prompt: str, schema: Type[T], agent_name: str) -> T:
        logger.info(json.dumps({
            "event": "agent_call_start",
            "agent": agent_name,
            "input": user_prompt[:2000],
        }, ensure_ascii=False))

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(json.dumps({
                "event": "agent_call_json_mode_unsupported_fallback",
                "agent": agent_name,
                "error": str(e),
            }, ensure_ascii=False))
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
        raw = response.choices[0].message.content or ""
        cleaned = _JSON_FENCE_RE.sub("", raw).strip()

        try:
            data = json.loads(cleaned)
            result = schema.model_validate(data)
        except Exception as e:
            logger.error(json.dumps({
                "event": "agent_call_parse_error",
                "agent": agent_name,
                "raw": raw[:2000],
                "error": str(e),
            }, ensure_ascii=False))
            raise

        logger.info(json.dumps({
            "event": "agent_call_done",
            "agent": agent_name,
            "output": result.model_dump(),
        }, ensure_ascii=False))
        return result
