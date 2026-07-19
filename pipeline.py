from typing import List, Union, Generator, Iterator
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        LLM_BASE_URL: str = "http://ollama:11434/v1"
        LLM_MODEL: str = "qwen2.5:7b-instruct"
        WHISPER_MODEL: str = "medium"

    def __init__(self):
        self.name = "MTBank Call Analytics"
        self.valves = self.Valves()

    async def on_startup(self):
        print(f"on_startup: {self.name}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown: {self.name}")
        pass

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        return (
            "✅ Pipeline подключена и работает.\n\n"
            f"Ты написал: \"{user_message}\"\n\n"
            "Загрузка и анализ аудио появятся на Дне 2."
        )
