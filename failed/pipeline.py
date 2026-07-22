"""
title: MTBank Call Analytics
author: mtbank-test-task
version: 0.3.0
description: >
  Транскрибация (faster-whisper) + диаризация + мультиагентный анализ
  (классификатор, качество, compliance, суммаризатор) через LangGraph.
requirements: faster-whisper, openai, langgraph, langchain-core
"""

import json
import logging
import os
import re
import sys
import tempfile
from typing import Generator, Iterator, List, Optional, Union

# OpenWebUI может грузить этот файл динамическим loader'ом, который не обязательно
# добавляет папку файла в sys.path — тогда `from agents...`/`from asr...` ниже
# упадут с ModuleNotFoundError, хотя эти пакеты физически лежат рядом.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from pydantic import BaseModel

from agents.llm_client import LLMClient
from agents.orchestrator import run_analysis
from asr.diarizer import (
    assign_speakers,
    detect_speech_islands,
)
from asr.transcriber import AudioConversionError, Transcriber, TranscriptionError

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pipeline")

AUDIO_URL_RE = re.compile(r"https?://\S+\.(?:wav|mp3|ogg)(?:\?\S*)?", re.IGNORECASE)


class Pipeline:
    class Valves(BaseModel):
        LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://ollama:11434/v1")
        LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
        WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")
        WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
        WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

    def __init__(self):
        self.name = "MTBank Call Analytics"
        self.valves = self.Valves()
        self.transcriber: Optional[Transcriber] = None
        self.llm: Optional[LLMClient] = None

    async def on_startup(self):
        logger.info(json.dumps({"event": "pipeline_startup", "name": self.name}))
        self.transcriber = Transcriber(
            model_size=self.valves.WHISPER_MODEL,
            device=self.valves.WHISPER_DEVICE,
            compute_type=self.valves.WHISPER_COMPUTE_TYPE,
        )
        self.llm = LLMClient(base_url=self.valves.LLM_BASE_URL, model=self.valves.LLM_MODEL)

    async def on_shutdown(self):
        logger.info(json.dumps({"event": "pipeline_shutdown", "name": self.name}))

    def _extract_audio_url(self, user_message: str) -> Optional[str]:
        match = AUDIO_URL_RE.search(user_message or "")
        return match.group(0) if match else None

    def _download_audio(self, url: str) -> str:
        ext = os.path.splitext(url.split("?")[0])[1] or ".wav"
        fd, path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as e:
            os.remove(path)
            raise RuntimeError(f"Не удалось скачать аудио по ссылке: {e}") from e
        return path

    def _format_response_markdown(self, segments: List[dict], analysis: dict) -> str:
        transcript_lines = ["| Время | Спикер | Текст |", "|---|---|---|"]
        for seg in segments:
            transcript_lines.append(
                f"| {seg['start']:.1f}-{seg['end']:.1f} | {seg['speaker']} | {seg['text']} |"
            )

        cls = analysis["classification"]
        q = analysis["quality_score"]
        c = analysis["compliance"]

        checklist_lines = "\n".join(
            f"  - {k}: {'✅' if v else '❌'}" for k, v in q["checklist"].items()
        )
        issues_lines = "\n".join(f"  - {issue}" for issue in c["issues"]) or "  - нет замечаний"
        action_items_lines = "\n".join(f"- {a}" for a in analysis["action_items"]) or "- нет"

        return (
            "\n".join(transcript_lines)
            + "\n\n### Классификация\n"
            + f"- Тема: **{cls['topic']}**, приоритет: **{cls['priority']}**\n"
            + f"\n### Качество ({q['total']}/100)\n{checklist_lines}\n"
            + f"\n### Compliance: {'✅ пройдено' if c['passed'] else '❌ есть замечания'}\n{issues_lines}\n"
            + f"\n### Резюме\n{analysis['summary']}\n"
            + f"\n### Action items\n{action_items_lines}"
        )

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        audio_url = self._extract_audio_url(user_message)

        if not audio_url:
            return (
                "Пришли ссылку на аудиофайл (WAV/MP3/OGG), например:\n"
                "`https://example.com/call.wav`"
            )

        local_path = None
        try:
            local_path = self._download_audio(audio_url)
            raw_segments = self.transcriber.transcribe(local_path)
            segments = assign_speakers(
    raw_segments,
    detect_speech_islands(local_path),
)
            if not segments:
                return "Не удалось распознать речь в этом файле (пустой результат)."

            analysis = run_analysis(segments, self.llm)
            return self._format_response_markdown(segments, analysis)
        except (AudioConversionError, TranscriptionError) as e:
            logger.error(json.dumps({"event": "pipe_asr_error", "error": str(e)}))
            return f"⚠️ Ошибка обработки аудио: {e}"
        except RuntimeError as e:
            logger.error(json.dumps({"event": "pipe_download_error", "error": str(e)}))
            return f"⚠️ {e}"
        except Exception as e:
            logger.error(json.dumps({"event": "pipe_unexpected_error", "error": str(e)}))
            return f"⚠️ Не удалось выполнить анализ: {e}"
        finally:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
