from unittest.mock import patch

import pytest

from src.study_assistant.agent import MAX_SUBQUERIES, analyze_query, decompose_query, rewrite_query
from src.study_assistant.orchestrator import process_student_query
from src.study_assistant.store import SearchResult


def test_router_validates_expected_intents():
    examples = [
        ("Hi", "casual_chat", False),
        ("I want to study physics", "study_guidance", False),
        ("What is electric flux?", "textbook_question", True),
        ("Explain that again", "follow_up_question", True),
        ("Calculate the electric field due to a point charge", "calculation", True),
    ]
    for question, intent, needs_retrieval in examples:
        with patch(
            "src.study_assistant.agent._groq_json",
            return_value={
                "intent": intent,
                "needs_retrieval": needs_retrieval,
                "standalone_query": question,
                "requires_calculation": intent == "calculation",
            },
        ):
            route = analyze_query(question, [])
        assert route["intent"] == intent
        assert route["needs_retrieval"] is needs_retrieval


def test_obvious_greeting_skips_llm_router():
    with patch("src.study_assistant.agent._groq_json") as groq_json:
        route = analyze_query("Hi!", [])

    groq_json.assert_not_called()
    assert route == {
        "intent": "casual_chat",
        "needs_retrieval": False,
        "standalone_query": "Hi!",
        "requires_calculation": False,
    }


def test_rewrite_query_resolves_follow_up_context():
    with patch(
        "src.study_assistant.agent._groq_json",
        return_value={"standalone_query": "Explain electric flux in relation to Gauss's law."},
    ):
        rewritten = rewrite_query(
            "Explain that again",
            [{"role": "user", "content": "What is electric flux?"}],
        )
    assert rewritten == "Explain electric flux in relation to Gauss's law."


def test_decompose_query_keeps_single_concept_as_one_query():
    with patch(
        "src.study_assistant.agent._groq_json",
        return_value={"decompose": False, "queries": ["What is electric flux?"]},
    ):
        result = decompose_query("What is electric flux?", [])
    assert result == {"decompose": False, "queries": ["What is electric flux?"]}


def test_decompose_query_caps_subqueries():
    with patch(
        "src.study_assistant.agent._groq_json",
        return_value={
            "decompose": True,
            "queries": [f"concept {index}" for index in range(MAX_SUBQUERIES + 2)],
        },
    ):
        result = decompose_query("Compare several concepts", [])
    assert result["decompose"] is True
    assert len(result["queries"]) == MAX_SUBQUERIES


def test_casual_orchestration_skips_qdrant():
    with (
        patch(
            "src.study_assistant.orchestrator.analyze_query",
            return_value={
                "intent": "casual_chat",
                "needs_retrieval": False,
                "standalone_query": "Hi",
                "requires_calculation": False,
            },
        ),
        patch(
            "src.study_assistant.orchestrator._conversational_answer",
            return_value="Hello! What would you like to study?",
        ),
        patch("src.study_assistant.orchestrator.qdrant_client") as qdrant,
    ):
        result = process_student_query("Hi", [])

    qdrant.assert_not_called()
    assert result["used_retrieval"] is False
    assert result["sources"] == []
    assert isinstance(result["debug"]["total_latency_ms"], float)


def test_textbook_orchestration_uses_retrieval_and_generation():
    fake_record = SearchResult("id", "text", 0.9, "Electric flux evidence.", 1, 1)
    with (
        patch(
            "src.study_assistant.orchestrator.analyze_query",
            return_value={
                "intent": "textbook_question",
                "needs_retrieval": True,
                "standalone_query": "What is electric flux?",
                "requires_calculation": False,
            },
        ),
        patch("src.study_assistant.orchestrator.qdrant_client", return_value="client"),
        patch(
            "src.study_assistant.orchestrator.decompose_query",
            return_value={"decompose": False, "queries": ["What is electric flux?"]},
        ),
        patch("src.study_assistant.orchestrator.search", return_value=[fake_record]) as search,
        patch(
            "src.study_assistant.orchestrator.generate_answer",
            return_value="Grounded answer",
        ) as generate,
    ):
        result = process_student_query("What is electric flux?", [])

    search.assert_called_once()
    generate.assert_called_once()
    assert result["used_retrieval"] is True
    assert result["sources"] == [fake_record]


def _textbook_route():
    return {
        "intent": "textbook_question",
        "needs_retrieval": True,
        "standalone_query": "What is electric flux?",
        "requires_calculation": False,
    }


def test_empty_first_retrieval_retries_and_generates_once():
    record = SearchResult("id", "text", 0.9, "Electric flux evidence.", 1, 1)
    with (
        patch("src.study_assistant.orchestrator.analyze_query", return_value=_textbook_route()),
        patch("src.study_assistant.orchestrator.qdrant_client", return_value="client"),
        patch(
            "src.study_assistant.orchestrator.decompose_query",
            return_value={"decompose": False, "queries": ["What is electric flux?"]},
        ),
        patch("src.study_assistant.orchestrator.search", side_effect=[[], [record]]) as search,
        patch("src.study_assistant.orchestrator.generate_answer", return_value="Answer") as generate,
    ):
        result = process_student_query("What is electric flux?", [])

    assert search.call_count == 2
    generate.assert_called_once()
    assert result["debug"]["retrieval_attempts"] == 2
    assert result["debug"]["evidence_sufficient"] is True


def test_two_empty_retrievals_return_insufficient_evidence_without_generation():
    with (
        patch("src.study_assistant.orchestrator.analyze_query", return_value=_textbook_route()),
        patch("src.study_assistant.orchestrator.qdrant_client", return_value="client"),
        patch(
            "src.study_assistant.orchestrator.decompose_query",
            return_value={"decompose": False, "queries": ["What is electric flux?"]},
        ),
        patch("src.study_assistant.orchestrator.search", side_effect=[[], None]) as search,
        patch("src.study_assistant.orchestrator.generate_answer") as generate,
    ):
        result = process_student_query("What is electric flux?", [])

    assert search.call_count == 2
    generate.assert_not_called()
    assert result["debug"]["retrieval_attempts"] == 2
    assert result["debug"]["evidence_sufficient"] is False
    assert "couldn't find enough relevant information" in result["answer"]


def test_malformed_records_are_treated_as_insufficient():
    with (
        patch("src.study_assistant.orchestrator.analyze_query", return_value=_textbook_route()),
        patch("src.study_assistant.orchestrator.qdrant_client", return_value="client"),
        patch(
            "src.study_assistant.orchestrator.decompose_query",
            return_value={"decompose": False, "queries": ["What is electric flux?"]},
        ),
        patch(
            "src.study_assistant.orchestrator.search",
            side_effect=[[{"unexpected": "value"}], [{"text": ""}]],
        ),
        patch("src.study_assistant.orchestrator.generate_answer") as generate,
    ):
        result = process_student_query("What is electric flux?", [])

    generate.assert_not_called()
    assert result["debug"]["evidence_sufficient"] is False


def test_comparison_decomposes_and_merges_duplicate_records():
    point_charge = SearchResult("point", "text", 0.9, "Point charge evidence.", 18, 18)
    dipole = SearchResult("dipole", "page", 0.8, "Dipole evidence.", 24, 24)
    with (
        patch("src.study_assistant.orchestrator.analyze_query", return_value=_textbook_route()),
        patch("src.study_assistant.orchestrator.qdrant_client", return_value="client"),
        patch(
            "src.study_assistant.orchestrator.decompose_query",
            return_value={
                "decompose": True,
                "queries": ["point charge field", "dipole field"],
            },
        ),
        patch(
            "src.study_assistant.orchestrator.search",
            side_effect=[[point_charge], [point_charge, dipole]],
        ) as search,
        patch(
            "src.study_assistant.orchestrator.generate_answer",
            return_value="Comparison answer",
        ) as generate,
    ):
        result = process_student_query("Compare point charge and dipole fields", [])

    assert search.call_count == 2
    generate.assert_called_once()
    assert result["sources"] == [point_charge, dipole]
    assert result["debug"]["decomposed"] is True
    assert result["debug"]["subqueries"] == ["point charge field", "dipole field"]


def test_follow_up_is_rewritten_before_decomposition():
    route = {
        "intent": "follow_up_question",
        "needs_retrieval": True,
        "standalone_query": "Explain that again",
        "requires_calculation": False,
    }
    record = SearchResult("id", "text", 0.9, "Electric field evidence.", 2, 2)
    with (
        patch("src.study_assistant.orchestrator.analyze_query", return_value=route),
        patch(
            "src.study_assistant.orchestrator.rewrite_query",
            return_value="Compare electric flux and electric field",
        ) as rewrite,
        patch(
            "src.study_assistant.orchestrator.decompose_query",
            return_value={"decompose": True, "queries": ["electric flux", "electric field"]},
        ),
        patch("src.study_assistant.orchestrator.qdrant_client", return_value="client"),
        patch("src.study_assistant.orchestrator.search", return_value=[record]),
        patch("src.study_assistant.orchestrator.generate_answer", return_value="Answer"),
    ):
        process_student_query("Explain that again", [{"role": "user", "content": "What is flux?"}])
    rewrite.assert_called_once()


def test_calculation_uses_deterministic_result_in_generation():
    record = SearchResult("id", "text", 0.9, "Electric field formula.", 11, 11)
    route = {
        "intent": "calculation",
        "needs_retrieval": True,
        "standalone_query": "Calculate electric field",
        "requires_calculation": True,
    }
    with (
        patch("src.study_assistant.orchestrator.analyze_query", return_value=route),
        patch(
            "src.study_assistant.orchestrator.decompose_query",
            return_value={"decompose": False, "queries": ["Calculate electric field"]},
        ),
        patch("src.study_assistant.orchestrator.qdrant_client", return_value="client"),
        patch("src.study_assistant.orchestrator.search", return_value=[record]),
        patch("src.study_assistant.orchestrator.generate_answer", return_value="Answer") as generate,
    ):
        result = process_student_query("A point charge of 2 μC is 3 m away. Calculate the electric field.", [])

    assert result["debug"]["calculation_used"] is True
    assert generate.call_args.args[4]["result"] == pytest.approx(1997.7777777777778)


def test_conceptual_question_does_not_use_calculator():
    record = SearchResult("id", "text", 0.9, "Electric field formula.", 11, 11)
    with (
        patch(
            "src.study_assistant.orchestrator.analyze_query",
            return_value=_textbook_route(),
        ),
        patch(
            "src.study_assistant.orchestrator.decompose_query",
            return_value={"decompose": False, "queries": ["What is electric flux?"]},
        ),
        patch("src.study_assistant.orchestrator.qdrant_client", return_value="client"),
        patch("src.study_assistant.orchestrator.search", return_value=[record]),
        patch("src.study_assistant.orchestrator.calculate_electric_field") as calculator,
        patch("src.study_assistant.orchestrator.generate_answer", return_value="Answer"),
    ):
        result = process_student_query("What is electric flux?", [])

    calculator.assert_not_called()
    assert result["debug"]["calculation_used"] is False
