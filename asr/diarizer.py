"""
Диаризация по реальным паузам в аудио (не по тайм-кодам Whisper).

ВАЖНО, найдено на практике: тайм-коды сегментов Whisper нельзя использовать
для поиска пауз между репликами — с vad_filter=True они сшиваются встык
(VAD физически вырезает тишину из аудио), а с vad_filter=False Whisper всё
равно режет текст на сегменты по своей внутренней логике (где заканчивается
"предложение"), а не по факту тишины в сигнале. В обоих случаях реальные
паузы между репликами в тайм-кодах сегментов не отражаются надёжно.

Поэтому паузы ищем сами, напрямую по амплитуде исходного аудио — независимо
от того, как Whisper решил порезать текст. Это даёт "острова" речи в реальном
времени файла; каждому сегменту Whisper присваивается спикер по номеру
острова, в который попадает середина сегмента, острова чередуются
Оператор/Клиент/Оператор/...

Ограничения (честно, для README):
  - всё ещё не различает говорящих по голосу/тембру — только по паузам
  - предполагает строгую очередность реплик без перебиваний/наложений
  - если острова определены неточно (шумная запись, слишком короткий порог
    тишины) — границы говорящих тоже поедут
  - для настоящей диаризации нужен pyannote.audio (см. бонусные задачи в ТЗ)
"""

import os
from typing import Dict, List, Tuple

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from asr.transcriber import convert_to_wav

# Низкий порог — чтобы поймать ВСЕ паузы, включая короткие естественные паузы
# TTS на запятых. Разделение "это просто пауза внутри фразы" от "это смена
# говорящего" происходит не здесь, а в _merge_islands_into_turns() ниже —
# фиксированный порог тишины для этого разделения не работает (проверено на
# практике: у edge-tts длительность внутренних пауз плывёт и пересекается с
# длительностью пауз между репликами).
BASE_SILENCE_LEN_MS = 300
SILENCE_THRESH_OFFSET_DB = 16

MIN_TURN_GAP_SEC = 0.5


def detect_speech_islands(audio_path: str) -> List[Tuple[float, float]]:
    """
    Возвращает список (start_sec, end_sec) островков речи, найденных по
    амплитуде исходного аудио, УЖЕ объединённых в предполагаемые реплики
    (см. _merge_islands_into_turns) — то есть соседние острова, разделённые
    короткой паузой (запятая, вдох), склеены в один; остаются только границы,
    похожие на реальную смену говорящего.
    """
    wav_path = convert_to_wav(audio_path)
    try:
        audio = AudioSegment.from_wav(wav_path)
        silence_thresh = audio.dBFS - SILENCE_THRESH_OFFSET_DB
        ranges_ms = detect_nonsilent(
            audio, min_silence_len=BASE_SILENCE_LEN_MS, silence_thresh=silence_thresh
        )
        raw_islands = [(start / 1000, end / 1000) for start, end in ranges_ms]
        return _merge_islands_into_turns(raw_islands)
    finally:
        os.remove(wav_path)


def _merge_islands_into_turns(
    islands: List[Tuple[float, float]],
    min_turn_gap_sec: float = MIN_TURN_GAP_SEC,
) -> List[Tuple[float, float]]:
    """
    islands найдены с низким порогом тишины (300мс) — среди них вперемешку и
    короткие внутрифразовые паузы, и настоящие паузы между репликами.

    Порог "что считать сменой говорящего" ищем не как фиксированную
    константу и не через медиану (медиана ломается, если реплик со сменой
    говорящего в файле БОЛЬШЕ, чем внутрифразовых пауз — тогда сама медиана
    уже "большая", и порог получается завышенным — поймано на локальном
    тесте до реального прогона), а через "естественный разрыв": сортируем
    все паузы файла и ищем, где между соседними по величине паузами самый
    большой скачок — это и есть граница между "короткими" и "длинными"
    паузами именно в этом файле.
    """
    if len(islands) <= 1:
        return islands

    gaps = [islands[i + 1][0] - islands[i][1] for i in range(len(islands) - 1)]
    threshold = _find_natural_gap_threshold(gaps, min_turn_gap_sec)

    merged = [islands[0]]
    for i, gap in enumerate(gaps):
        next_island = islands[i + 1]
        if gap >= threshold:
            merged.append(next_island)
        else:
            last_start, _ = merged[-1]
            merged[-1] = (last_start, next_island[1])
    return merged


def _find_natural_gap_threshold(gaps: List[float], min_turn_gap_sec: float) -> float:
    sorted_gaps = sorted(gaps)
    if len(sorted_gaps) == 1:
        return max(min_turn_gap_sec, sorted_gaps[0])

    biggest_jump = 0.0
    split_value = sorted_gaps[-1]
    for i in range(len(sorted_gaps) - 1):
        jump = sorted_gaps[i + 1] - sorted_gaps[i]
        if jump > biggest_jump:
            biggest_jump = jump
            split_value = (sorted_gaps[i] + sorted_gaps[i + 1]) / 2
    return max(min_turn_gap_sec, split_value)


def _find_island_index(t: float, islands: List[Tuple[float, float]]) -> int:
    for i, (start, end) in enumerate(islands):
        if start <= t <= end:
            return i
    # середина сегмента не попала точно ни в один остров (пограничный случай
    # из-за неточности тайм-кодов Whisper) — берём ближайший по расстоянию
    return min(
        range(len(islands)),
        key=lambda i: min(abs(t - islands[i][0]), abs(t - islands[i][1])),
    )


def assign_speakers(
    segments: List[Dict],
    islands: List[Tuple[float, float]],
    first_speaker: str = "Оператор",
) -> List[Dict]:
    """
    segments: сегменты Whisper [{"start": float, "end": float, "text": str}, ...]
    islands: острова речи из detect_speech_islands(), в хронологическом порядке.

    Предположение: звонок в контакт-центре всегда начинает Оператор
    (приветствие) — поэтому first_speaker по умолчанию "Оператор", и острова
    считаются строго чередующимися репликами без перебиваний.
    """
    if not segments:
        return []
    if not islands:
        return [{**seg, "speaker": first_speaker} for seg in segments]

    other_speaker = "Клиент" if first_speaker == "Оператор" else "Оператор"
    result = []
    for seg in segments:
        midpoint = (seg["start"] + seg["end"]) / 2
        island_idx = _find_island_index(midpoint, islands)
        speaker = first_speaker if island_idx % 2 == 0 else other_speaker
        result.append({**seg, "speaker": speaker})
    return result
