"""
Считает WER (jiwer) для всех файлов в test_data/ против эталонных .txt.

Запускать ВНУТРИ контейнера pipelines (там уже есть faster-whisper, GPU, ffmpeg):

    sudo docker compose exec pipelines python3 test_data/compute_wer.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jiwer import wer

from asr.transcriber import Transcriber

TEST_DATA_DIR = Path(__file__).resolve().parent

# (аудиофайл, эталонный транскрипт) — 8kHz-версия сравнивается с тем же
# эталоном, что и оригинал call_1_kredity, т.к. это тот же диалог.
FILES = [
    ("call_1_kredity.mp3", "call_1_kredity.txt"),
    ("call_1_kredity_8khz.wav", "call_1_kredity.txt"),
    ("call_2_karty.mp3", "call_2_karty.txt"),
    ("call_3_perevody.mp3", "call_3_perevody.txt"),
    ("call_4_zhaloby.mp3", "call_4_zhaloby.txt"),
    ("call_5_drugoe.mp3", "call_5_drugoe.txt"),
]


def load_reference_text(txt_path: Path) -> str:
    lines = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # строки вида "Оператор: текст" -> берём только текст после спикера
        parts = line.split(": ", 1)
        lines.append(parts[1] if len(parts) == 2 else line)
    return " ".join(lines)


def main():
    transcriber = Transcriber(
        model_size=os.getenv("WHISPER_MODEL", "small"),
        device=os.getenv("WHISPER_DEVICE", "cuda"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
    )

    rows = []
    for audio_name, ref_name in FILES:
        audio_path = TEST_DATA_DIR / audio_name
        ref_path = TEST_DATA_DIR / ref_name
        if not audio_path.exists() or not ref_path.exists():
            print(f"пропускаю {audio_name}: файл не найден (сгенерируй через generate_test_data.py)")
            continue

        reference = load_reference_text(ref_path)
        raw_segments = transcriber.transcribe(str(audio_path))
        hypothesis = " ".join(seg["text"] for seg in raw_segments)

        score = wer(reference, hypothesis)
        rows.append((audio_name, len(raw_segments), score))
        print(f"{audio_name}: WER = {score:.3f}")

    print("\n| Файл | Сегментов | WER |")
    print("|---|---|---|")
    for name, n_seg, score in rows:
        print(f"| {name} | {n_seg} | {score:.3f} |")


if __name__ == "__main__":
    main()
