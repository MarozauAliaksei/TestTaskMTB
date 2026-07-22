from agents.classifier import ClassificationResult
from agents.compliance import ComplianceResult
from agents.orchestrator import run_analysis
from agents.quality import QualityChecklist, QualityResult
from agents.summarizer import SummaryResult
from asr.diarizer import assign_speakers


class FakeLLM:
    def call_structured(self, system_prompt, user_prompt, schema, agent_name):
        if agent_name == "classifier":
            return ClassificationResult(topic="кредиты", priority="medium")
        if agent_name == "quality":
            return QualityResult(
                total=75,
                checklist=QualityChecklist(
                    greeting=True, need_detection=True, solution_provided=True, farewell=False
                ),
            )
        if agent_name == "compliance":
            return ComplianceResult(passed=True, issues=[])
        if agent_name == "summarizer":
            return SummaryResult(
                summary="Клиент обратился по вопросу кредита.",
                action_items=["Перезвонить клиенту"],
            )
        raise ValueError(f"unexpected agent: {agent_name}")


def test_run_analysis_combines_all_agents():
    segments = [
        {"speaker": "Оператор", "start": 0.0, "end": 2.0, "text": "Добрый день"},
        {"speaker": "Клиент", "start": 2.5, "end": 5.0, "text": "Хочу узнать про кредит"},
    ]
    result = run_analysis(segments, FakeLLM())

    assert result["classification"]["topic"] == "кредиты"
    assert result["quality_score"]["total"] == 75
    assert result["compliance"]["passed"] is True
    assert "кредит" in result["summary"]
    assert result["action_items"] == ["Перезвонить клиенту"]


def test_assign_speakers_uses_islands():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Добрый день, банк, меня зовут Анна"},
        {"start": 2.1, "end": 3.0, "text": "чем могу помочь"},
        {"start": 4.5, "end": 6.0, "text": "здравствуйте хочу узнать про кредит"},
    ]
    # два острова речи, найденные независимо по аудио: первый покрывает первые
    # два сегмента (это один говорящий), второй — третий сегмент (сменился говорящий)
    islands = [(0.0, 3.2), (4.3, 6.2)]
    result = assign_speakers(segments, islands)

    assert result[0]["speaker"] == "Оператор"
    assert result[1]["speaker"] == "Оператор"
    assert result[2]["speaker"] == "Клиент"


def test_assign_speakers_handles_empty_segments():
    assert assign_speakers([], [(0.0, 1.0)]) == []


def test_assign_speakers_handles_empty_islands():
    segments = [{"start": 0.0, "end": 1.0, "text": "тест"}]
    result = assign_speakers(segments, [])
    assert result[0]["speaker"] == "Оператор"
