"""Deterministic unit and integration tests; no test uses network or a real API key."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import Mock, call

from main import classify_record
import pytest
import requests

import eval as evaluation
import main

ROOT = Path(__file__).resolve().parent


def valid_model_json(article: str | None, status: str = "classified") -> str:
    return json.dumps({
        "article": article,
        "status": status,
        "explanation": "Reasoning grounded in the provided target phrase.",
        "evidence": ["The supplied sentence contains the target phrase."],
    })


class StubClient:
    def __init__(self, content: str | Exception = "") -> None:
        self.content = content
        self.messages: list[list[dict[str, str]]] = []
        self.calls = 0

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.calls += 1
        self.messages.append([dict(message) for message in messages])
        if isinstance(self.content, Exception):
            raise self.content
        return self.content


class FakeResponse:
    def __init__(self, status_code: int, body: Any = None, json_error: Exception | None = None) -> None:
        self.status_code = status_code
        self.body = body
        self.json_error = json_error

    def json(self) -> Any:
        if self.json_error is not None:
            raise self.json_error
        return self.body


def sample_kb() -> dict[str, Any]:
    return {
        "rules": [
            {"id": "always", "always": True, "trigger_terms": [], "guidance": "Classify only the target; abstain if scope is unclear."},
            {"id": "plural", "trigger_terms": ["dogs", "students"], "guidance": "Bare plural may be zero article."},
            {"id": "mass", "trigger_terms": ["music", "information"], "guidance": "Mass noun can be bare, but check determiners."},
        ]
    }


def test_valid_llm_classification_response() -> None:
    client = StubClient(valid_model_json("zero"))
    result = main.classify_record(
        {"id": "x1", "text": "Dogs are great.", "target_phrase": "Dogs", "context": ""},
        mode="agent", knowledge_base=sample_kb(), client=client,
    )
    assert result["article"] == "zero"
    assert result["status"] == "classified"
    assert result["source"] == "llm"
    assert result["id"] == "x1"
    assert client.calls == 1
    assert "retrieved_rules" in client.messages[0][-1]["content"]


def test_deterministic_overt_article_fast_path_does_not_call_llm() -> None:
    client = StubClient(valid_model_json("zero"))
    result = main.classify_record(
        {"id": "x2", "text": "I saw a dog.", "target_phrase": "a dog"},
        mode="agent", knowledge_base=sample_kb(), client=client,
    )
    assert result["article"] == "indefinite"
    assert result["source"] == "deterministic"
    assert client.calls == 0


@pytest.mark.parametrize(
    "record,expected_field",
    [
        ({"id": "x", "text": "", "target_phrase": "the book"}, "text"),
        ({"id": "x", "target_phrase": "the book"}, "text"),
        ({"id": "x", "text": "The book is here."}, "target_phrase"),
        ({"id": "", "text": "The book is here.", "target_phrase": "the book"}, "id"),
        ({"id": "x", "text": "The book is here.", "target_phrase": "the book", "context": 12}, "context"),
    ],
)
def test_empty_missing_or_invalid_input_fields(record: dict[str, Any], expected_field: str) -> None:
    result = main.classify_record(record, mode="agent", knowledge_base=sample_kb(), client=StubClient(valid_model_json("definite")))
    assert result["status"] == "error"
    assert result["article"] is None
    assert result["error"]["code"] == "invalid_input"
    assert expected_field in result["error"]["message"]


def test_different_target_phrases_in_one_sentence_get_different_labels() -> None:
    sentence = "I saw a dog near the house."
    indefinite = main.classify_record({"id": "one", "text": sentence, "target_phrase": "a dog"}, mode="agent", knowledge_base=sample_kb())
    definite = main.classify_record({"id": "two", "text": sentence, "target_phrase": "the house"}, mode="agent", knowledge_base=sample_kb())
    assert indefinite["article"] == "indefinite"
    assert definite["article"] == "definite"
    assert indefinite["id"] == "one" and definite["id"] == "two"


def test_ambiguous_target_not_found_does_not_call_llm() -> None:
    client = StubClient(valid_model_json("zero"))
    result = main.classify_record(
        {"id": "missing", "text": "I saw a dog near the house.", "target_phrase": "the building"},
        mode="agent", knowledge_base=sample_kb(), client=client,
    )
    assert result["status"] == "ambiguous"
    assert result["article"] is None
    assert result["error"] is None
    assert client.calls == 0


def test_incomplete_target_span_is_not_automatically_zero() -> None:
    result = main.classify_record(
        {"id": "partial", "text": "The dog was barking loudly.", "target_phrase": "dog"},
        mode="agent", knowledge_base=sample_kb(), client=StubClient(valid_model_json("zero")),
    )
    assert result["status"] == "ambiguous"
    assert result["article"] is None
    assert result["source"] == "deterministic"


def test_ambiguous_llm_response_is_preserved() -> None:
    content = valid_model_json(None, "ambiguous")
    client = StubClient(content)
    result = main.classify_record(
        {"id": "amb", "text": "My brother repaired the car.", "target_phrase": "my brother"},
        mode="agent", knowledge_base=sample_kb(), client=client,
    )
    assert result["status"] == "ambiguous"
    assert result["article"] is None
    assert result["error"] is None


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"article": "maybe", "status": "classified", "explanation": "x", "evidence": ["y"]}),
        json.dumps({"article": None, "status": "classified", "explanation": "x", "evidence": ["y"]}),
        json.dumps({"article": "zero", "status": "ambiguous", "explanation": "x", "evidence": ["y"]}),
        json.dumps({"article": "zero", "status": "classified", "explanation": "", "evidence": ["y"]}),
        json.dumps({"article": ["zero"], "status": "classified", "explanation": "x", "evidence": ["y"]}),
        json.dumps({"article": "zero", "status": ["classified"], "explanation": "x", "evidence": ["y"]}),
        json.dumps({"article": "zero", "status": "classified", "explanation": "x", "evidence": "not a list"}),
        json.dumps({"article": "zero", "status": "classified", "explanation": "x", "evidence": ["y"], "extra": 1}),
    ],
)
def test_malformed_json_and_invalid_output_schemas_are_rejected(content: str) -> None:
    with pytest.raises(main.ModelOutputError):
        main.parse_model_output(content)
    result = main.classify_record(
        {"id": "invalid-output", "text": "Dogs are great.", "target_phrase": "Dogs"},
        mode="baseline", client=StubClient(content),
    )
    assert result["status"] == "error"
    assert result["article"] is None
    assert result["error"]["code"] == "invalid_model_output"
    assert result["fallback_used"] is True


def test_knowledge_lookup_returns_only_matching_rules() -> None:
    kb = sample_kb()
    result = main.knowledge_lookup("Dogs are great.", "Dogs", kb)
    assert [rule["id"] for rule in result] == ["always", "plural"]
    assert main.knowledge_lookup("No matching noun.", "something", {"rules": [{"id": "x", "trigger_terms": ["music"]}]}) == []
    assert main.knowledge_lookup("Dogs", "Dogs", kb, max_rules=0) == []


def test_build_messages_distinguishes_baseline_from_agent() -> None:
    record = {"id": "m", "text": "Dogs are great.", "target_phrase": "Dogs", "context": "We discussed dogs earlier."}
    rules = [{"id": "R", "guidance": "A matching local rule."}]
    baseline = main.build_messages(record, mode="baseline", retrieved_rules=rules)
    agent = main.build_messages(record, mode="agent", retrieved_rules=rules)
    baseline_last = json.loads(baseline[-1]["content"])
    agent_last = json.loads(agent[-1]["content"])
    assert "context" not in baseline_last and "retrieved_rules" not in baseline_last
    assert agent_last["context"] == "We discussed dogs earlier."
    assert agent_last["retrieved_rules"] == rules


@pytest.mark.parametrize("retry_status", [429, 500, 503, 599])
def test_retryable_http_statuses_retry_then_succeed(monkeypatch: pytest.MonkeyPatch, retry_status: int) -> None:
    success_body = {"choices": [{"message": {"content": valid_model_json("zero")}}]}
    post = Mock(side_effect=[FakeResponse(retry_status, {}), FakeResponse(200, success_body)])
    sleep = Mock()
    monkeypatch.setattr(main.requests, "post", post)
    monkeypatch.setattr(main.time, "sleep", sleep)
    client = main.OpenRouterClient(api_key="not-a-real-key", timeout_seconds=12, retry_delay_seconds=2)
    content = client.complete([])
    assert json.loads(content)["article"] == "zero"
    assert post.call_count == 2
    assert sleep.call_args_list == [call(2)]
    assert post.call_args.kwargs["timeout"] == (5.0, 12.0)


def test_connection_errors_and_timeouts_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"choices": [{"message": {"content": valid_model_json("definite")}}]}
    outcomes: list[Any] = [
        requests.exceptions.ConnectionError("offline"),
        requests.exceptions.Timeout("timed out"),
        FakeResponse(200, body),
    ]
    calls: list[dict[str, Any]] = []

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    sleeps: list[float] = []
    monkeypatch.setattr(main.requests, "post", fake_post)
    monkeypatch.setattr(main.time, "sleep", lambda seconds: sleeps.append(seconds))
    content = main.OpenRouterClient(api_key="secret-placeholder", timeout_seconds=9).complete([])
    assert json.loads(content)["article"] == "definite"
    assert len(calls) == 3
    assert sleeps == [2.0, 2.0]
    assert calls[0]["timeout"] == (5.0, 9.0)
    assert "Authorization" in calls[0]["headers"]


def test_non_retryable_http_4xx_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(400, {"error": "bad request"})

    monkeypatch.setattr(main.requests, "post", fake_post)
    monkeypatch.setattr(main.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(main.APIError) as captured:
        main.OpenRouterClient(api_key="placeholder").complete([])
    assert captured.value.code == "http_error"
    assert calls == 1
    assert sleeps == []


def test_retry_count_is_bounded_and_no_sleep_after_final_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(503, {})

    monkeypatch.setattr(main.requests, "post", fake_post)
    monkeypatch.setattr(main.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(main.APIError) as captured:
        main.OpenRouterClient(api_key="placeholder", max_attempts=9).complete([])
    assert captured.value.code == "http_retry_exhausted"
    assert calls == 3  # hard cap, even if caller requests more
    assert sleeps == [2.0, 2.0]


def test_malformed_api_json_and_bad_envelope_are_reported_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(200, json_error=ValueError("broken json"))

    monkeypatch.setattr(main.requests, "post", fake_post)
    with pytest.raises(main.APIError) as captured:
        main.OpenRouterClient(api_key="placeholder").complete([])
    assert captured.value.code == "invalid_api_json"
    assert calls == 1

    monkeypatch.setattr(main.requests, "post", lambda *args, **kwargs: FakeResponse(200, {"choices": []}))
    with pytest.raises(main.APIError) as captured2:
        main.OpenRouterClient(api_key="placeholder").complete([])
    assert captured2.value.code == "invalid_api_response"


def test_missing_api_key_fails_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(
        main.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("network must not be called"),
    )

    with pytest.raises(main.APIError) as captured:
        main.OpenRouterClient(api_key="").complete([])

    assert captured.value.code == "missing_api_key"


def test_other_requests_transport_errors_become_non_retryable_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        raise requests.exceptions.InvalidURL("bad URL")

    sleep = Mock()
    monkeypatch.setattr(main.requests, "post", fake_post)
    monkeypatch.setattr(main.time, "sleep", sleep)
    with pytest.raises(main.APIError) as captured:
        main.OpenRouterClient(api_key="placeholder").complete([])
    assert captured.value.code == "transport_error"
    assert captured.value.retryable is False
    assert calls == 1
    sleep.assert_not_called()


def test_jsonl_output_preserves_record_order_and_identifiers(tmp_path: Path) -> None:
    results = [
        main.classify_record({"id": "second", "text": "I saw a dog.", "target_phrase": "a dog"}, mode="agent", knowledge_base=sample_kb()),
        main.classify_record({"id": "first", "text": "The dog barked.", "target_phrase": "the dog"}, mode="agent", knowledge_base=sample_kb()),
    ]
    path = tmp_path / "nested" / "results.jsonl"
    main.write_jsonl(results, path)
    read_back = main.read_jsonl(path)
    assert [row["id"] for row in read_back] == ["second", "first"]
    assert [row["article"] for row in read_back] == ["indefinite", "definite"]
    assert all({"id", "text", "target_phrase", "article", "status", "explanation", "evidence", "error"} <= set(row) for row in read_back)


def test_batch_duplicate_id_and_failed_record_do_not_stop_later_records() -> None:
    records = [
        {"id": "same", "text": "I saw a dog.", "target_phrase": "a dog"},
        {"id": "same", "text": "The house is large.", "target_phrase": "the house"},
        {"id": "last", "text": "Dogs are great.", "target_phrase": "Dogs"},
    ]
    client = StubClient(valid_model_json("zero"))
    results = main.process_records(records, mode="agent", knowledge_base=sample_kb(), client=client)
    assert [item["id"] for item in results] == ["same", "same", "last"]
    assert [item["status"] for item in results] == ["classified", "error", "classified"]
    assert results[1]["error"]["code"] == "duplicate_id"


def test_api_failure_on_one_record_does_not_stop_remaining_batch() -> None:
    class FailOnceClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
            self.calls += 1
            if self.calls == 1:
                raise main.APIError("http_retry_exhausted", "mock exhausted retries", retryable=True)
            return valid_model_json("zero")

    records = [
        {"id": "api-fail", "text": "Dogs are great.", "target_phrase": "Dogs"},
        {"id": "next-record", "text": "Birds migrate south.", "target_phrase": "Birds"},
    ]
    client = FailOnceClient()
    results = main.process_records(records, mode="baseline", client=client)
    assert [result["id"] for result in results] == ["api-fail", "next-record"]
    assert results[0]["status"] == "error"
    assert results[0]["article"] is None and results[0]["fallback_used"] is True
    assert results[1]["status"] == "classified" and results[1]["article"] == "zero"
    assert client.calls == 2


def test_end_to_end_batch_with_external_calls_mocked(tmp_path: Path) -> None:
    records = [
        {"id": "e1", "text": "I saw a dog.", "target_phrase": "a dog"},
        {"id": "e2", "text": "Dogs are great.", "target_phrase": "Dogs"},
        {"id": "e3", "text": "The dog is here.", "target_phrase": ""},
        {"id": "e4", "text": "I spoke to her.", "target_phrase": "the house"},
    ]
    client = StubClient(valid_model_json("zero"))
    results = main.process_records(records, mode="agent", knowledge_base=sample_kb(), client=client)
    output_path = tmp_path / "results.jsonl"
    main.write_jsonl(results, output_path)
    loaded = main.read_jsonl(output_path)
    assert [row["id"] for row in loaded] == ["e1", "e2", "e3", "e4"]
    assert loaded[0]["article"] == "indefinite" and loaded[0]["source"] == "deterministic"
    assert loaded[1]["article"] == "zero" and loaded[1]["source"] == "llm"
    assert loaded[2]["status"] == "error" and loaded[2]["error"]["code"] == "invalid_input"
    assert loaded[3]["status"] == "ambiguous"
    assert loaded[3]["article"] is None
    assert client.calls == 1


def _gold(id_: str, article: str | None, status: str = "classified", phenomenon: str = "sample") -> dict[str, Any]:
    return {"id": id_, "expected_article": article, "expected_status": status, "phenomenon": phenomenon, "eval_scope": "linguistic", "split": "test"}


def _pred(id_: str, article: str | None, status: str = "classified", fallback: bool = False) -> dict[str, Any]:
    return {"id": id_, "article": article, "status": status, "fallback_used": fallback, "source": "fallback" if fallback else "llm"}


def test_evaluation_metrics_correct_incorrect_ambiguous_missing_and_operational_failure() -> None:
    gold = [
        _gold("g1", "definite", phenomenon="definite"),
        _gold("g2", "indefinite", phenomenon="indefinite"),
        _gold("g3", None, status="ambiguous", phenomenon="ambiguous_gold"),
        _gold("g4", "zero", phenomenon="zero"),  # deliberately missing prediction
        _gold("g5", "zero", phenomenon="api_failure"),
    ]
    predictions = [
        _pred("g1", "definite"),
        _pred("g2", "definite"),  # incorrect
        _pred("g3", None, status="ambiguous"),
        _pred("g5", None, status="error", fallback=True),
    ]
    metrics = evaluation.calculate_metrics(gold, predictions)
    assert metrics["n_linguistic_examples"] == 5
    assert metrics["classifiable_gold_n"] == 4
    assert metrics["correct_classifiable"] == 1
    assert metrics["accuracy_on_classifiable"] == pytest.approx(0.25)
    assert metrics["coverage"] == pytest.approx(2 / 5)
    assert metrics["ambiguity_rate"] == pytest.approx(1 / 5)
    assert metrics["fallback_rate"] == pytest.approx(1 / 5)
    assert metrics["operational_failure_rate"] == pytest.approx(2 / 5)
    assert metrics["missing_prediction_n"] == 1
    assert metrics["ambiguity_recall_on_gold_ambiguous"] == pytest.approx(1.0)
    assert metrics["per_class"]["indefinite"]["false_negative"] == 1
    assert "accuracy_and_per_class" in metrics["denominators"]
    assert metrics["by_phenomenon"]["definite"]["accuracy"] == pytest.approx(1.0)
    assert metrics["error_analysis"]["outcome_counts"]["incorrect_label"] == 1
    assert metrics["error_analysis"]["outcome_counts"]["missing_prediction"] == 1
    assert metrics["error_analysis"]["outcome_counts"]["operational_failure"] == 1
    assert any("adjudicated counterexamples" in item for item in metrics["error_analysis"]["suggested_next_steps"])
    assert {item["id"]: item["outcome"] for item in metrics["example_diagnostics"]}["g2"] == "incorrect_label"


def test_evaluation_treats_abstention_as_incorrect_for_classifiable_gold() -> None:
    gold = [_gold("g1", "zero")]
    predictions = [_pred("g1", None, status="ambiguous")]
    metrics = evaluation.calculate_metrics(gold, predictions)
    assert metrics["accuracy_on_classifiable"] == pytest.approx(0.0)
    assert metrics["coverage"] == pytest.approx(0.0)
    assert metrics["ambiguity_rate"] == pytest.approx(1.0)
    assert metrics["error_analysis"]["outcome_counts"]["abstained_on_classifiable_gold"] == 1


def test_compare_metrics_reports_baseline_agent_and_deltas() -> None:
    gold = [_gold("x", "zero")]
    baseline = [_pred("x", "definite")]
    agent = [_pred("x", "zero")]
    compared = evaluation.compare_metrics(gold, baseline, agent)
    assert compared["baseline"]["accuracy_on_classifiable"] == 0.0
    assert compared["agent"]["accuracy_on_classifiable"] == 1.0
    assert compared["agent_minus_baseline"]["accuracy_on_classifiable"] == 1.0


def test_run_comparison_runs_both_configs_on_same_held_out_examples() -> None:
    gold_rows = [
        {"id": "test-dog", "text": "Dogs are great.", "target_phrase": "Dogs", "context": "", "expected_article": "zero", "expected_status": "classified", "phenomenon": "generic_bare_plural", "split": "test", "eval_scope": "linguistic"},
        {"id": "dev-only", "text": "The dog barked.", "target_phrase": "the dog", "context": "", "expected_article": "definite", "expected_status": "classified", "phenomenon": "clear_definite", "split": "dev", "eval_scope": "linguistic"},
    ]

    class ModeAwareClient:
        def __init__(self) -> None:
            self.modes: list[str] = []

        def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
            mode = "agent" if "fixed English-article classification pipeline" in messages[0]["content"] else "baseline"
            self.modes.append(mode)
            label = "zero" if mode == "agent" else "definite"
            return valid_model_json(label)

    client = ModeAwareClient()
    baseline, agent, metrics = evaluation.run_comparison(gold_rows, split="test", knowledge_base=sample_kb(), client=client)
    assert [row["id"] for row in baseline] == ["test-dog"]
    assert [row["id"] for row in agent] == ["test-dog"]
    assert metrics["baseline"]["accuracy_on_classifiable"] == 0.0
    assert metrics["agent"]["accuracy_on_classifiable"] == 1.0
    assert client.modes == ["baseline", "agent"]


def test_read_golden_separates_linguistic_and_robustness_rows() -> None:
    rows = evaluation.read_golden(ROOT / "golden.csv")
    assert sum(row["split"] == "dev" and row["eval_scope"] == "linguistic" for row in rows) == 12
    assert sum(row["split"] == "test" and row["eval_scope"] == "linguistic" for row in rows) == 26
    assert sum(row["eval_scope"] == "robustness" for row in rows) == 4
    assert any(row["phenomenon"] == "api_failure_fixture" for row in rows)


def test_school_phrase_accepts_model_classification() -> None:
    from main import classify_record

    client = StubClient(valid_model_json("zero"))

    result = classify_record(
        {
            "id": "I07",
            "text": "The children went to school after breakfast.",
            "target_phrase": "school",
        },
        mode="agent",
        client=client,
    )

    assert result["article"] == "zero"
    assert result["status"] == "classified"
    assert result["source"] == "llm"

def test_bare_singular_count_noun_rejects_unsupported_evidence() -> None:
    bad_response = json.dumps({
        "article": "zero",
        "status": "classified",
        "explanation": "The target has an overt article.",
        "evidence": ["Overt article: 'the'."],
    })
    client = StubClient(bad_response)

    result = main.classify_record(
        {
            "id": "I06",
            "text": "I went to bank yesterday.",
            "target_phrase": "bank",
        },
        mode="agent",
        knowledge_base=sample_kb(),
        client=client,
    )

    assert result["status"] == "error"
    assert result["article"] is None
    assert result["error"]["code"] == "invalid_model_output"
    assert result["fallback_used"] is True


def test_validator_rejects_indefinite_without_overt_article() -> None:
    parsed = {
        "article": "indefinite",
        "status": "classified",
        "explanation": "The target has an indefinite article.",
        "evidence": [],
    }

    with pytest.raises(main.ModelOutputError):
        main.validate_model_answer(parsed, "Dogs are great.", "Dogs")


def test_validator_rejects_definite_without_overt_article() -> None:
    parsed = {
        "article": "definite",
        "status": "classified",
        "explanation": "The target has a definite article.",
        "evidence": [],
    }

    with pytest.raises(main.ModelOutputError):
        main.validate_model_answer(
            parsed,
            "Mount Everest attracts climbers.",
            "Mount Everest",
        )



def test_alternative_determiner_returns_ambiguous():
    class NeverCallClient:
        def complete(self, messages):
            raise AssertionError(
                "The model should not be called for this input."
            )

    record = {
    "id": "test-alternative-determiner",
    "text": "I found some information online.",
    "target_phrase": "some information",
    }

    result = classify_record(
        record,
        mode="agent",
        client=NeverCallClient(),
    )
    assert result.get("error") is None, f"Unexpected error result: {result!r}"
    print("ALTERNATIVE DETERMINER RESULT:", result, flush=True)

    assert result["article"] is None
    assert result["status"] == "ambiguous"
    assert result["source"] == "deterministic"




def test_bare_plural_still_reaches_agent():
    record = {
    "id": "test-bare-plural",
    "text": "Cats often sleep during the day.",
    "target_phrase": "Cats",
    }

    response = json.dumps({
        "article": "zero",
        "status": "classified",
        "explanation": (
            "The target is a bare plural noun with no overt article "
            "or competing determiner."
        ),
        "evidence": [
            "Target phrase: 'Cats'",
            "The target is a bare plural.",
            "No overt article appears before the target.",
        ],
    })

    result = classify_record(
        record,
        mode="agent",
        client=StubClient(response),
    )
    print("BARE PLURAL RESULT:", result, flush=True)

    assert result["article"] == "zero"
    assert result["status"] == "classified"
    assert result["source"] == "llm"
