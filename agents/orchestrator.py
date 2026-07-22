from typing import Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.classifier import ClassificationResult, classify
from agents.compliance import ComplianceResult, check_compliance
from agents.llm_client import LLMClient
from agents.quality import QualityResult, evaluate_quality
from agents.summarizer import SummaryResult, summarize


class AnalysisState(TypedDict, total=False):
    transcript_text: str
    classification: ClassificationResult
    quality: QualityResult
    compliance: ComplianceResult
    summary: SummaryResult


def format_transcript_for_llm(segments: List[Dict]) -> str:
    return "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)


def build_graph(llm: LLMClient):
    graph = StateGraph(AnalysisState)

    graph.add_node("classify", lambda s: {"classification": classify(s["transcript_text"], llm)})
    graph.add_node("quality", lambda s: {"quality": evaluate_quality(s["transcript_text"], llm)})
    graph.add_node("compliance", lambda s: {"compliance": check_compliance(s["transcript_text"], llm)})
    graph.add_node("summarize", lambda s: {"summary": summarize(s["transcript_text"], llm)})

    # classify/quality/compliance читают только исходный транскрипт и не зависят
    # друг от друга — запускаем их параллельно из START, а не цепочкой.
    graph.add_edge(START, "classify")
    graph.add_edge(START, "quality")
    graph.add_edge(START, "compliance")

    # summarize стартует только после того, как все три предыдущих узла завершились
    graph.add_edge("classify", "summarize")
    graph.add_edge("quality", "summarize")
    graph.add_edge("compliance", "summarize")

    graph.add_edge("summarize", END)

    return graph.compile()


def run_analysis(segments: List[Dict], llm: LLMClient) -> Dict:
    transcript_text = format_transcript_for_llm(segments)
    app = build_graph(llm)
    result = app.invoke({"transcript_text": transcript_text})

    return {
        "classification": result["classification"].model_dump(),
        "quality_score": result["quality"].model_dump(),
        "compliance": result["compliance"].model_dump(),
        "summary": result["summary"].summary,
        "action_items": result["summary"].action_items,
    }
