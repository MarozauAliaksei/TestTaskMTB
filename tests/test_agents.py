from agents.classifier import ClassificationResult, classify
from agents.compliance import ComplianceResult, check_compliance
from agents.quality import QualityChecklist, QualityResult, evaluate_quality
from agents.summarizer import SummaryResult, summarize


class FakeLLM:
    def __init__(self, result):
        self.result = result
        self.called_with = None

    def call_structured(self, system_prompt, user_prompt, schema, agent_name):
        self.called_with = (system_prompt, user_prompt, agent_name)
        return self.result


def test_classify_returns_parsed_result():
    fake = FakeLLM(ClassificationResult(topic="кредиты", priority="medium"))
    result = classify("тестовый транскрипт", fake)
    assert result.topic == "кредиты"
    assert result.priority == "medium"
    assert fake.called_with[2] == "classifier"


def test_evaluate_quality_returns_parsed_result():
    fake = FakeLLM(QualityResult(
        total=80,
        checklist=QualityChecklist(greeting=True, need_detection=True, solution_provided=True, farewell=False),
    ))
    result = evaluate_quality("транскрипт", fake)
    assert result.total == 80
    assert result.checklist.farewell is False
    assert fake.called_with[2] == "quality"


def test_check_compliance_returns_parsed_result():
    fake = FakeLLM(ComplianceResult(passed=False, issues=["гарантия одобрения кредита"]))
    result = check_compliance("транскрипт", fake)
    assert result.passed is False
    assert len(result.issues) == 1
    assert fake.called_with[2] == "compliance"


def test_compliance_result_coerces_dict_shaped_issues():
    # Реальный кейс, встреченный при тестировании: qwen2.5:3b вернула
    # issues как список объектов вместо списка строк.
    result = ComplianceResult.model_validate({
        "passed": False,
        "issues": [{"issue": "Запрещённые обещания о гарантии одобрения кредита"}],
    })
    assert result.issues == ["Запрещённые обещания о гарантии одобрения кредита"]


def test_summary_result_coerces_dict_shaped_action_items():
    result = SummaryResult.model_validate({
        "summary": "тест",
        "action_items": [{"action": "Перезвонить клиенту"}],
    })
    assert result.action_items == ["Перезвонить клиенту"]


def test_summarize_returns_parsed_result():
    fake = FakeLLM(SummaryResult(summary="Клиент спросил про кредит.", action_items=["Отправить КП"]))
    result = summarize("транскрипт", fake)
    assert "кредит" in result.summary
    assert result.action_items == ["Отправить КП"]
    assert fake.called_with[2] == "summarizer"
