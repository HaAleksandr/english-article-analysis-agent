"""Evaluation utilities for the English Article Analysis Agent.

Accuracy is measured only on gold examples annotated as classifiable, with
abstentions counted as incorrect. Coverage is reported on all linguistic examples
in the selected split, including gold-ambiguous cases. Operational failures and
ambiguous predictions are reported separately. Robustness fixtures are excluded
from linguistic quality metrics because API outage behavior is not a gold article
label; those conditions are exercised with mocked tests.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from main import LABELS, load_knowledge_base, process_records, read_jsonl, write_jsonl

LOGGER = logging.getLogger("article_agent.eval")


def read_golden(path: str | Path) -> list[dict[str, Any]]:
    """Read a golden CSV, preserving annotation metadata and blank labels."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Golden CSV is empty or has no header row.")
        rows = [dict(row) for row in reader]
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        row_id = row.get("id", "")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"Golden CSV row {row_number} has no id.")
        if row_id in seen:
            raise ValueError(f"Golden CSV contains duplicate id {row_id!r}.")
        seen.add(row_id)
        row["expected_article"] = (row.get("expected_article") or "").strip() or None
        row["expected_status"] = (row.get("expected_status") or "classified").strip() or "classified"
        row["eval_scope"] = (row.get("eval_scope") or "linguistic").strip() or "linguistic"
        row["split"] = (row.get("split") or "test").strip() or "test"
        if row["expected_article"] is not None and row["expected_article"] not in LABELS:
            raise ValueError(f"Golden CSV row {row_number} has unsupported expected_article.")
        if row["expected_status"] not in {"classified", "ambiguous", "error"}:
            raise ValueError(f"Golden CSV row {row_number} has unsupported expected_status.")
    return rows


def _classified_label(prediction: Mapping[str, Any] | None) -> str | None:
    if not isinstance(prediction, Mapping):
        return None
    article = prediction.get("article")
    if prediction.get("status") == "classified" and isinstance(article, str) and article in LABELS:
        return article
    return None


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0 if precision == 0.0 or recall == 0.0 else None
    return 2 * precision * recall / (precision + recall)


def calculate_metrics(
    gold_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Calculate classification, coverage, uncertainty, and failure metrics.

    Predictions are matched by ``id``. A missing prediction is not silently dropped:
    for classifiable gold rows it counts as incorrect; for rates it counts as a
    missing operational result. Only rows with ``eval_scope=linguistic`` and gold
    status ``classified`` or ``ambiguous`` enter linguistic denominators.
    """
    prediction_by_id: dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        prediction_id = prediction.get("id")
        if isinstance(prediction_id, str) and prediction_id not in prediction_by_id:
            prediction_by_id[prediction_id] = prediction

    examples = [
        row for row in gold_rows
        if row.get("eval_scope", "linguistic") == "linguistic"
        and row.get("expected_status", "classified") in {"classified", "ambiguous"}
    ]
    total = len(examples)
    classifiable = [
        row for row in examples
        if row.get("expected_status", "classified") == "classified"
        and row.get("expected_article") in LABELS
    ]
    correct = 0
    for row in classifiable:
        pred = prediction_by_id.get(str(row.get("id")))
        if _classified_label(pred) == row.get("expected_article"):
            correct += 1

    end_to_end_correct = 0

    for row in examples:
        pred = prediction_by_id.get(str(row.get("id")))
        if pred is None:
            continue

        expected_status = row.get("expected_status", "classified")
        predicted_status = pred.get("status")

        if expected_status == "ambiguous":
            if predicted_status == "ambiguous":
                end_to_end_correct += 1
        elif expected_status == "classified":
            if (
                predicted_status == "classified"
                and _classified_label(pred) == row.get("expected_article")
            ):
                end_to_end_correct += 1

    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for label in sorted(LABELS):
        tp = fp = fn = support = 0
        for row in classifiable:
            gold_label = row.get("expected_article")
            pred_label = _classified_label(prediction_by_id.get(str(row.get("id"))))
            if gold_label == label:
                support += 1
                if pred_label == label:
                    tp += 1
                else:
                    fn += 1
            elif pred_label == label:
                fp += 1
        precision = _divide(tp, tp + fp)
        recall = _divide(tp, tp + fn)
        f1 = _f1(precision, recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
        if f1 is not None:
            f1_values.append(f1)

    classified_predictions = sum(
        _classified_label(prediction_by_id.get(str(row.get("id")))) is not None
        for row in examples
    )
    ambiguous_predictions = sum(
        prediction_by_id.get(str(row.get("id")), {}).get("status") == "ambiguous"
        for row in examples
    )
    fallback_predictions = sum(
        bool(prediction_by_id.get(str(row.get("id")), {}).get("fallback_used", False))
        or prediction_by_id.get(str(row.get("id")), {}).get("source") == "fallback"
        for row in examples
    )
    operational_failures = sum(
        str(row.get("id")) not in prediction_by_id
        or prediction_by_id[str(row.get("id"))].get("status") == "error"
        for row in examples
    )
    missing_predictions = sum(str(row.get("id")) not in prediction_by_id for row in examples)
    gold_ambiguous = [row for row in examples if row.get("expected_status") == "ambiguous"]
    ambiguity_true_positive = sum(
        prediction_by_id.get(str(row.get("id")), {}).get("status") == "ambiguous"
        for row in gold_ambiguous
    )

    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in examples:
        by_category[str(row.get("phenomenon") or "unspecified")].append(row)
    category_metrics: dict[str, dict[str, Any]] = {}
    for category, category_rows in sorted(by_category.items()):
        category_classifiable = [
            row for row in category_rows
            if row.get("expected_status") == "classified" and row.get("expected_article") in LABELS
        ]
        category_correct = sum(
            _classified_label(prediction_by_id.get(str(row.get("id")))) == row.get("expected_article")
            for row in category_classifiable
        )
        category_covered = sum(
            _classified_label(prediction_by_id.get(str(row.get("id")))) is not None
            for row in category_rows
        )
        category_metrics[category] = {
            "n": len(category_rows),
            "classifiable_n": len(category_classifiable),
            "correct": category_correct,
            "accuracy": _divide(category_correct, len(category_classifiable)),
            "coverage": _divide(category_covered, len(category_rows)),
        }

    # Record-level diagnostics turn the aggregate report into an actionable error
    # analysis. These outcomes are based on labels/statuses, not the model's prose.
    diagnostics: list[dict[str, Any]] = []
    for row in examples:
        row_id = str(row.get("id"))
        prediction = prediction_by_id.get(row_id)
        predicted_status = str(prediction.get("status")) if prediction is not None else "missing"
        predicted_label = _classified_label(prediction)
        expected_status = str(row.get("expected_status", "classified"))
        expected_label = row.get("expected_article")
        if expected_status == "ambiguous":
            if predicted_status == "ambiguous":
                outcome = "correct_abstention"
            elif prediction is None:
                outcome = "missing_prediction"
            elif predicted_status == "error":
                outcome = "operational_failure"
            elif predicted_label is not None:
                outcome = "confident_prediction_on_ambiguous_gold"
            else:
                outcome = "invalid_prediction"
        elif predicted_label == expected_label:
            outcome = "correct"
        elif prediction is None:
            outcome = "missing_prediction"
        elif predicted_status == "error":
            outcome = "operational_failure"
        elif predicted_status == "ambiguous":
            outcome = "abstained_on_classifiable_gold"
        elif predicted_label is not None:
            outcome = "incorrect_label"
        else:
            outcome = "invalid_prediction"
        error_value = prediction.get("error") if prediction is not None else None
        error_code = error_value.get("code") if isinstance(error_value, Mapping) else None
        diagnostics.append({
            "id": row_id,
            "phenomenon": row.get("phenomenon") or "unspecified",
            "expected_article": expected_label,
            "expected_status": expected_status,
            "predicted_article": predicted_label,
            "predicted_status": predicted_status,
            "outcome": outcome,
            "error_code": error_code,
        })

    outcome_counts = Counter(item["outcome"] for item in diagnostics)
    failure_outcomes = {
        "confident_prediction_on_ambiguous_gold",
        "abstained_on_classifiable_gold",
        "incorrect_label",
        "invalid_prediction",
        "missing_prediction",
        "operational_failure",
    }
    failure_by_phenomenon: dict[str, int] = Counter(
        str(item["phenomenon"]) for item in diagnostics if item["outcome"] in failure_outcomes
    )
    suggested_next_steps: list[str] = []
    if outcome_counts["incorrect_label"]:
        suggested_next_steps.append("Review incorrect labels and add adjudicated counterexamples to the development split before changing prompts or rules.")
    if outcome_counts["abstained_on_classifiable_gold"]:
        suggested_next_steps.append("Review abstentions on classifiable cases; clarify when bare plural or mass-noun constructions are licensed.")
    if outcome_counts["confident_prediction_on_ambiguous_gold"]:
        suggested_next_steps.append("Add scope and incomplete-target counterexamples, and strengthen instructions to abstain when the target is underspecified.")
    if outcome_counts["operational_failure"] or outcome_counts["missing_prediction"] or outcome_counts["invalid_prediction"]:
        suggested_next_steps.append("Inspect error codes and prediction-file completeness; resolve key, quota, retry, transport, or schema failures separately from linguistic errors.")

    return {
        "n_linguistic_examples": total,
        "classifiable_gold_n": len(classifiable),
        "correct_classifiable": correct,
        "accuracy_on_classifiable": _divide(correct, len(classifiable)),
        "per_class": per_class,
        "macro_f1": _divide(round(sum(f1_values), 12), len(f1_values)),
        "coverage": _divide(classified_predictions, total),
        "classified_prediction_n": classified_predictions,
        "ambiguity_rate": _divide(ambiguous_predictions, total),
        "ambiguous_prediction_n": ambiguous_predictions,
        "gold_ambiguous_n": len(gold_ambiguous),
        "ambiguity_recall_on_gold_ambiguous": _divide(ambiguity_true_positive, len(gold_ambiguous)),
        "fallback_rate": _divide(fallback_predictions, total),
        "fallback_n": fallback_predictions,
        "operational_failure_rate": _divide(operational_failures, total),
        "operational_failure_n": operational_failures,
        "missing_prediction_n": missing_predictions,
        "example_diagnostics": diagnostics,
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_accuracy": _divide(end_to_end_correct, total),
        "error_analysis": {
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "failed_examples_by_phenomenon": dict(sorted(failure_by_phenomenon.items())),
            "suggested_next_steps": suggested_next_steps,
        },
        "denominators": {
            "accuracy_and_per_class": "gold examples with expected_status=classified and a supported expected_article; abstentions/errors/missing predictions count as incorrect",
            "coverage_ambiguity_fallback_operational_failure": "all linguistic examples in the selected split, including gold-ambiguous examples",
            "macro_f1": "arithmetic mean of defined per-class F1 scores for definite, indefinite, and zero; each applicable class has equal weight",
            "ambiguity_recall_on_gold_ambiguous": "gold linguistic examples whose expected_status=ambiguous",
        },
        "by_phenomenon": category_metrics,
    }


def compare_metrics(
    gold_rows: Sequence[Mapping[str, Any]],
    baseline_predictions: Sequence[Mapping[str, Any]],
    agent_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize baseline and agent on the same gold rows with metric deltas."""
    baseline = calculate_metrics(gold_rows, baseline_predictions)
    agent = calculate_metrics(gold_rows, agent_predictions)
    deltas: dict[str, float | None] = {}
    for metric in ("accuracy_on_classifiable", "macro_f1", "coverage", "ambiguity_rate", "fallback_rate", "operational_failure_rate"):
        left = baseline[metric]
        right = agent[metric]
        deltas[metric] = round(right - left, 6) if isinstance(left, (float, int)) and isinstance(right, (float, int)) else None
    return {
        "evaluation": "baseline_vs_agent",
        "comparison_split": next((str(row.get("split")) for row in gold_rows if row.get("eval_scope", "linguistic") == "linguistic"), None),
        "baseline": baseline,
        "agent": agent,
        "agent_minus_baseline": deltas,
    }


def _input_records(gold_rows: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("id"),
            "text": row.get("text"),
            "target_phrase": row.get("target_phrase"),
            "context": row.get("context") or "",
        }
        for row in gold_rows
        if row.get("split") == split and row.get("eval_scope", "linguistic") == "linguistic"
    ]


def run_comparison(
    gold_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    knowledge_base: Mapping[str, Any],
    client: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run baseline and agent on exactly the same linguistic examples."""
    selected = [row for row in gold_rows if row.get("split") == split and row.get("eval_scope", "linguistic") == "linguistic"]
    if not selected:
        raise ValueError(f"No linguistic examples found for split={split!r}.")
    records = _input_records(gold_rows, split)
    # The same transport configuration is used for both modes; each record is still
    # independently handled by process_records. A supplied client is a test seam.
    baseline_results = process_records(records, mode="baseline", knowledge_base=None, client=client)
    agent_results = process_records(records, mode="agent", knowledge_base=knowledge_base, client=client)
    metrics = compare_metrics(selected, baseline_results, agent_results)
    return baseline_results, agent_results, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and agent configurations on a fixed gold split.")
    parser.add_argument("--golden", default="golden.csv", help="Golden dataset CSV path.")
    parser.add_argument("--knowledge-base", default="knowledge_base.json", help="Local rules JSON path.")
    parser.add_argument("--split", choices=("dev", "test"), default="test", help="Dataset split to evaluate.")
    parser.add_argument("--output-dir", default="outputs/evaluation", help="Directory for prediction and metric artifacts.")
    parser.add_argument("--baseline-predictions", help="Optional previously saved baseline JSONL; avoids API calls.")
    parser.add_argument("--agent-predictions", help="Optional previously saved agent JSONL; avoids API calls.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        gold_rows = read_golden(args.golden)
        selected = [row for row in gold_rows if row.get("split") == args.split and row.get("eval_scope", "linguistic") == "linguistic"]
        if not selected:
            raise ValueError(f"No linguistic examples found for split={args.split!r}.")
        if bool(args.baseline_predictions) != bool(args.agent_predictions):
            raise ValueError("Pass both --baseline-predictions and --agent-predictions, or neither.")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.baseline_predictions and args.agent_predictions:
            baseline_results = read_jsonl(args.baseline_predictions)
            agent_results = read_jsonl(args.agent_predictions)
            metrics = compare_metrics(selected, baseline_results, agent_results)
        else:
            knowledge_base = load_knowledge_base(args.knowledge_base)
            baseline_results, agent_results, metrics = run_comparison(
                gold_rows, split=args.split, knowledge_base=knowledge_base
            )
        write_jsonl(baseline_results, output_dir / "baseline_predictions.jsonl")
        write_jsonl(agent_results, output_dir / "agent_predictions.jsonl")
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 2

    print(json.dumps({
        "split": args.split,
        "n_linguistic_examples": metrics["baseline"]["n_linguistic_examples"],
        "baseline_accuracy": metrics["baseline"]["accuracy_on_classifiable"],
        "baseline_coverage": metrics["baseline"]["coverage"],
        "agent_accuracy": metrics["agent"]["accuracy_on_classifiable"],
        "agent_coverage": metrics["agent"]["coverage"],
        "metrics_path": str(output_dir / "metrics.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
