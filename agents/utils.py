"""
Небольшие модели (у нас qwen2.5:3b) иногда не строго следуют запрошенной
JSON-схеме, даже с response_format=json_object. Характерный пример, реально
встреченный при тестировании: вместо {"issues": ["..."]} модель вернула
{"issues": [{"issue": "..."}]} — обернула строку в объект.

Эта функция приводит такие отклонения к простому списку строк вместо того,
чтобы валиться с ошибкой валидации Pydantic.
"""

from typing import Any, List


def coerce_string_list(value: Any) -> Any:
    if not isinstance(value, list):
        return value

    result: List[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            string_value = next((v for v in item.values() if isinstance(v, str)), None)
            result.append(string_value if string_value is not None else str(item))
        else:
            result.append(str(item))
    return result
