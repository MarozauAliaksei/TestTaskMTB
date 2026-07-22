"""
Обёртка над faster-whisper.

Отвечает только за одно: путь к аудиофайлу -> список сегментов с таймкодами и текстом.
Разделение на говорящих (Оператор/Клиент) — отдельный шаг, см. diarizer.py.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

from faster_whisper import WhisperModel

logger = logging.getLogger("asr.transcriber")

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg"}


class AudioConversionError(Exception):
    """Аудиофайл не удалось привести к нужному формату (плохой файл, неподдерживаемый формат, ffmpeg упал)."""


class TranscriptionError(Exception):
    """Whisper не смог обработать файл (битые данные, OOM на GPU и т.п.)."""


def convert_to_wav(input_path: str) -> str:
    """
    Приводит любой поддерживаемый входной формат к 16kHz mono WAV через ffmpeg.
    Это нужно по двум причинам:
      1. Единообразный вход для Whisper независимо от исходного формата/частоты дискретизации
      2. Явная поддержка телефонного качества (8kHz) — ffmpeg сам передискретизирует
    Возвращает путь к временному wav-файлу. Вызывающий код отвечает за удаление.
    """
    ext = Path(input_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioConversionError(
            f"Неподдерживаемый формат '{ext}'. Поддерживаются: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000",
        "-f", "wav", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        raise AudioConversionError(f"ffmpeg завершился с ошибкой: {stderr}") from e
    except subprocess.TimeoutExpired as e:
        raise AudioConversionError("Конвертация ffmpeg превысила таймаут (120с)") from e

    return out_path


class Transcriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ru",
    ):
        logger.info(json.dumps({
            "event": "whisper_model_loading",
            "model": model_size,
            "device": device,
            "compute_type": compute_type,
        }))
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            # Частый случай на слабых GPU: не хватило VRAM под float16 — пробуем откатиться на CPU,
            # чтобы прототип хотя бы не падал полностью, но громко логируем деградацию.
            logger.warning(json.dumps({
                "event": "whisper_gpu_load_failed_fallback_cpu",
                "error": str(e),
            }))
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.language = language

    def transcribe(self, audio_path: str) -> List[Dict]:
        """
        Возвращает список сегментов: [{"start": float, "end": float, "text": str}, ...]
        Без поля speaker — это добавляется отдельно в diarizer.assign_speakers().
        """
        if not os.path.exists(audio_path):
            raise TranscriptionError(f"Файл не найден: {audio_path}")

        wav_path = convert_to_wav(audio_path)
        try:
            segments_iter, info = self.model.transcribe(
                wav_path,
                language=self.language,
                # ВАЖНО: vad_filter здесь намеренно ВЫКЛЮЧЕН.
                # vad_filter=True физически вырезает тишину из аудио перед
                # распознаванием и сшивает тайм-коды сегментов встык — реальные
                # паузы между репликами при этом теряются (проверено на практике:
                # с vad_filter=True все сегменты шли строго встык, end одного ==
                # start следующего, независимо от настоящих пауз в файле). Наша
                # диаризация в diarizer.py опирается именно на реальные паузы в
                # тайм-кодах, поэтому вместо VAD используем "сырую" сегментацию
                # Whisper по исходной временной шкале.
                # Компромисс: на реальных шумных записях (не наши чистые TTS-
                # тестовые файлы) без VAD выше риск галлюцинаций Whisper на
                # тишине/шуме — это тот случай, когда стоит вернуться к VAD и
                # искать другой способ диаризации (например, отдельный анализ
                # пауз по амплитуде исходного аудио, не через Whisper).
                vad_filter=False,
            )
            result = [
                {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                }
                for seg in segments_iter
                if seg.text.strip()
            ]
            logger.info(json.dumps({
                "event": "transcription_done",
                "segments": len(result),
                "language": info.language,
                "duration_sec": round(info.duration, 2),
            }))
            return result
        except Exception as e:
            logger.error(json.dumps({"event": "transcription_failed", "error": str(e)}))
            raise TranscriptionError(str(e)) from e
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
