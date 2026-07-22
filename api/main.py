import logging
import os
import tempfile
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from agents.llm_client import LLMClient
from agents.orchestrator import run_analysis
from asr.diarizer import assign_speakers, detect_speech_islands
from asr.transcriber import AudioConversionError, Transcriber, TranscriptionError

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="MTBank Call Analytics API")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://ollama:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

transcriber: Optional[Transcriber] = None
llm_client: Optional[LLMClient] = None


@app.on_event("startup")
def startup():
    global transcriber, llm_client
    transcriber = Transcriber(model_size=WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    llm_client = LLMClient(base_url=LLM_BASE_URL, model=LLM_MODEL, api_key=LLM_API_KEY)


async def _save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    return path


def _download_url(url: str) -> str:
    ext = os.path.splitext(url.split("?")[0])[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return path


def _process(path: str) -> dict:
    raw_segments = transcriber.transcribe(path)
    islands = detect_speech_islands(path)
    segments = assign_speakers(raw_segments, islands)
    analysis = run_analysis(segments, llm_client)
    return {
        "transcript": segments,
        "classification": analysis["classification"],
        "quality_score": analysis["quality_score"],
        "compliance": analysis["compliance"],
        "summary": analysis["summary"],
        "action_items": analysis["action_items"],
    }


@app.post("/analyze")
async def analyze(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)):
    if not file and not url:
        raise HTTPException(status_code=400, detail="Нужно передать file или url")

    local_path = None
    try:
        local_path = await _save_upload(file) if file else _download_url(url)
        return _process(local_path)
    except (AudioConversionError, TranscriptionError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Не удалось скачать файл по url: {e}")
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail="Внутренняя ошибка обработки")
    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)


@app.get("/health")
def health():
    return {"status": "ok"}
