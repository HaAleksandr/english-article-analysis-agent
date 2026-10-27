"""English Article Analysis Agent: classification, transport, and JSONL I/O.

The pipeline is deliberately fixed and small. It is an LLM application with a
local knowledge tool, not an autonomous agent. The deterministic fast path only
classifies a target noun phrase when it is found in the sentence and explicitly
starts with ``a``, ``an``, or ``the``. Bare phrases are not labelled ``zero``
merely because no article appears in the supplied string.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests
from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger("article_agent")
LABELS = {"definite", "indefinite", "zero"}
MODEL_STATUSES = {"classified", "ambiguous"}
RESULT_STATUSES = {"classified", "ambiguous", "error"}
DEFAULT_MODEL = "meta-llama/llama-3-8b-instruct:free"
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_API_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2.0

# JSON returned by the model must have exactly these four keys.
MODEL_OUTPUT_KEYS = {"article", "status", "explanation", "evidence"}
PRECEDING_DETERMINERS = {
    "a", "an", "the", "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "some", "any", "each", "every", "no",
    "many", "few", "several", "much", "all", "both", "either", "neither",
}

BASELINE_SYSTEM_PROMPT = """You classify the article usage of one specified target noun phrase in an English sentence.
Classify only the target phrase, not the whole sentence. Output exactly one JSON object
with these keys: article, status, explanation, evidence. article must be "definite",
"indefinite", "zero", or null. status must be "classified" or "ambiguous".
Use status="classified" only when article is one of the three labels. Use
status="ambiguous" and article=null when the target is outside the task's scope or the
evidence is insufficient. explanation must be a concise string; evidence must be a list
of strings. Do not include Markdown fences or any text outside JSON.

Scope: definite and indefinite refer to overt English articles "the" and "a/an".
Use zero only for a well-formed noun phrase with no overt article and no competing
determiner, where a bare noun construction is licensed (for example, a generic bare
plural or mass noun, or an ordinary proper name such as Paris). A phrase with a
possessive, demonstrative, quantifier, pronoun, or an unclear/incomplete target span is
not automatically zero; abstain when outside scope or unclear. Bare plurals can also
refer to specific entities and still have zero article. Do not confuse reference with
article form, invent context, or infer that missing a/an/the proves zero.

Examples:
Input: {"text":"I saw a dog near the house.","target_phrase":"a dog"}
Output: {"article":"indefinite","status":"classified","explanation":"The target noun phrase begins with the overt indefinite article 'a'.","evidence":["Target phrase: 'a dog'","Overt article: 'a'"]}
Input: {"text":"I saw a dog near the house.","target_phrase":"the house"}
Output: {"article":"definite","status":"classified","explanation":"The target noun phrase begins with the overt definite article 'the'.","evidence":["Target phrase: 'the house'","Overt article: 'the'"]}
Input: {"text":"Dogs are great.","target_phrase":"Dogs"}
Output: {"article":"zero","status":"classified","explanation":"'Dogs' is a bare plural with no overt article; the generic reading is licensed here.","evidence":["The target is a bare plural","No overt article or competing determiner appears in the target phrase"]}
Input: {"text":"I spoke to her.","target_phrase":"her"}
Output: {"article":null,"status":"ambiguous","explanation":"The target is a pronoun rather than a noun phrase whose article usage is being classified.","evidence":["Target phrase is a pronoun"]}
"""

AGENT_SYSTEM_PROMPT = """You are the linguistic analysis component of a fixed English-article classification pipeline.
Analyze ONLY the specified target noun phrase, not the sentence as a whole. Use the supplied
sentence, optional prior context, and retrieved local rules. Apply grammatical-form evidence
separately from discourse interpretation: an overt 'the' signals definite article form even
when context is sparse, while a bare plural can be generic or specific and still have zero
article. Do not infer that absence of a/an/the automatically means zero. Possessives,
demonstratives, quantifiers, pronouns, incomplete spans, and singular count nouns without a
licensing determiner may be outside scope or malformed; mark genuinely unresolved cases
ambiguous rather than inventing a confident answer. Proper nouns and institutional expressions
can vary by lexical convention, dialect, and meaning; use only evidence supplied by the input
and rules. For conventional institutional expressions such as "go to school", "go to bed",
and "at work", identify the construction as a whole. In "The children went to school",
"school" is a singular noun in an institutional expression, not a bare plural. Never invent
a rule or claim that supplied context establishes more than it does.

Return exactly one JSON object with exactly these keys: article, status, explanation, evidence.
article must be "definite", "indefinite", "zero", or null. status must be "classified" or
"ambiguous". If status is "classified", article must be one of the three labels. If status is
"ambiguous", article must be null. explanation must be a concise non-empty string. evidence
must be a list of strings. Do not include Markdown fences or text outside JSON.
"""


FEW_SHOT_EXAMPLES: list[tuple[dict[str, str], dict[str, Any]]] = [
    (
        {"text": "I saw a dog near the house.", "target_phrase": "a dog"},
        {
            "article": "indefinite",
            "status": "classified",
            "explanation": "The target noun phrase begins with the overt indefinite article 'a'.",
            "evidence": ["Target phrase: 'a dog'", "Overt article: 'a'"],
        },
    ),
    (
        {"text": "I saw a dog near the house.", "target_phrase": "the house"},
        {
            "article": "definite",
            "status": "classified",
            "explanation": "The target noun phrase begins with the overt definite article 'the'.",
            "evidence": ["Target phrase: 'the house'", "Overt article: 'the'"],
        },
    ),
    (
        {"text": "Dogs are great.", "target_phrase": "Dogs"},
        {
            "article": "zero",
            "status": "classified",
            "explanation": "The target is a bare plural and has no overt article or competing determiner.",
            "evidence": [
                "The target is a bare plural",
                "No overt article appears in the target phrase",
            ],
        },
    ),
    (
        {"text": "I spoke to her.", "target_phrase": "her"},
        {
            "article": None,
            "status": "ambiguous",
            "explanation": "The target is a pronoun, not a noun phrase within the intended scope.",
            "evidence": ["Target phrase is a pronoun"],
        },
    ),
    (
        {"text": "Mount Fuji is visible from the train.", "target_phrase": "Mount Fuji"},
        {
            "article": "zero",
            "status": "classified",
            "explanation": "This proper name is conventionally used without an article.",
            "evidence": [
                "The target is a proper name",
                "No overt article appears in the target phrase",
            ],
        },
    ),
    (
        {
            "text": "The United States has many national parks.",
            "target_phrase": "The United States",
        },
        {
            "article": "definite",
            "status": "classified",
            "explanation": "The target phrase begins with the overt definite article 'The'.",
            "evidence": [
                "Target phrase: 'The United States'",
                "Overt article: 'The'",
            ],
        },
    ),
    (
        {"text": "I spoke to him after lunch.", "target_phrase": "him"},
        {
            "article": None,
            "status": "ambiguous",
            "explanation": "The target is a pronoun rather than a noun phrase within the intended scope.",
            "evidence": ["Target phrase is a pronoun"],
        },
    ),
]


class APIError(RuntimeError):
    """A sanitized API/transport failure safe to include in a result record."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class ModelOutputError(ValueError):
    """Raised when model content is not valid under the required output schema."""


def _result_base(record: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the identifying/input fields even when a record is invalid."""
    return {
        "id": record.get("id"),
        "text": record.get("text"),
        "target_phrase": record.get("target_phrase"),
    }


def _make_result(
    record: Mapping[str, Any],
    *,
    article: str | None,
    status: str,
    explanation: str,
    evidence: Sequence[str] | None = None,
    source: str = "none",
    error: dict[str, Any] | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    """Build the stable, serializable result contract used by CLI and evaluation."""
    return {
        **_result_base(record),
        "article": article,
        "status": status,
        "explanation": explanation,
        "evidence": list(evidence or []),
        "source": source,
        "error": error,
        "fallback_used": fallback_used,
    }


def validate_record(record: Mapping[str, Any]) -> str | None:
    """Return a human-readable validation error, or ``None`` for a valid record.

    Contract: ``id``, ``text`` and ``target_phrase`` must be non-empty strings;
    optional ``context`` must be a string or ``None``. Extra fields are ignored.
    Batch-level ID uniqueness is checked by :func:`process_records`.
    """
    for field in ("id", "text", "target_phrase"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            return f"Field '{field}' must be a non-empty string."
    context = record.get("context", "")
    if context is not None and not isinstance(context, str):
        return "Field 'context' must be a string or null."
    return None


def _normalized_target_pattern(target_phrase: str) -> re.Pattern[str]:
    phrase = " ".join(target_phrase.strip().split())
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def target_phrase_occurs(text: str, target_phrase: str) -> bool:
    """Check whether the requested phrase occurs as a contiguous token span.

    Matching is case-insensitive and tolerates whitespace differences, but it does
    not infer an omitted article or perform syntactic/lemmatic reconstruction.
    """
    if not text.strip() or not target_phrase.strip():
        return False
    return _normalized_target_pattern(target_phrase).search(text) is not None


def target_span_has_preceding_determiner(text: str, target_phrase: str) -> bool:
    """Detect a likely incomplete target span immediately following a determiner.

    This deliberately conservative check catches inputs such as target ``dog`` in
    sentence ``The dog barked``. It does not do full noun-phrase parsing, so more
    complex incomplete spans (for example ``dog`` in ``the old dog``) still require
    the LLM/knowledge rules to abstain.
    """
    match = _normalized_target_pattern(target_phrase).search(text)
    if match is None:
        return False
    prefix_tokens = re.findall(r"[\w'-]+", text[:match.start()])
    return bool(prefix_tokens and prefix_tokens[-1].casefold() in PRECEDING_DETERMINERS)


def deterministic_article(text: str, target_phrase: str) -> str | None:
    """Return a label only for a verified target with an overt leading article.

    Returning ``None`` means ``use a later strategy``; it does not mean zero article.
    This intentionally narrow rule avoids treating article absence as proof of zero.
    """
    if not target_phrase_occurs(text, target_phrase):
        return None
    phrase = target_phrase.strip()
    match = re.match(r"^(a|an|the)\b", phrase, flags=re.IGNORECASE)
    if match is None:
        return None
    # A determiner alone is not a complete noun phrase for this task.
    if len(re.findall(r"\b[\w'-]+\b", phrase)) < 2:
        return None
    article = match.group(1).casefold()
    return "definite" if article == "the" else "indefinite"



def knowledge_lookup(
    text: str,
    target_phrase: str,
    knowledge_base: Mapping[str, Any],
    *,
    max_rules: int = 6,
) -> list[dict[str, Any]]:
    """Retrieve relevant local linguistic rules for one classification request.

    Input contract: sentence, target phrase, and a parsed JSON object containing a
    ``rules`` list. Each rule may declare ``always: true`` or a list of string
    ``trigger_terms``. A rule matches if any trigger occurs case-insensitively in the
    sentence or target phrase. The function returns at most ``max_rules`` matching
    rule dictionaries in knowledge-base order; it makes no classification itself.
    """
    if not isinstance(text, str) or not isinstance(target_phrase, str):
        return []
    raw_rules = knowledge_base.get("rules", [])
    if not isinstance(raw_rules, list) or max_rules <= 0:
        return []
    search_text = f"{text} {target_phrase}".casefold()
    matches: list[dict[str, Any]] = []
    for rule in raw_rules:
        if not isinstance(rule, dict):
            continue
        triggers = rule.get("trigger_terms", [])
        if not isinstance(triggers, list) or not all(isinstance(item, str) for item in triggers):
            continue
        is_match = rule.get("always") is True or any(
            term.casefold() in search_text for term in triggers if term
        )
        if is_match:
            matches.append(dict(rule))
            if len(matches) >= max(1, max_rules):
                break
    return matches


def load_knowledge_base(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the curated local knowledge base."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise ValueError("Knowledge base must be a JSON object with a 'rules' list.")
    return data


def parse_model_output(content: str) -> dict[str, Any]:
    """Parse a JSON string and strictly validate every model output field."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ModelOutputError("Model response is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ModelOutputError("Model response must be a JSON object.")
    if set(parsed.keys()) != MODEL_OUTPUT_KEYS:
        missing = sorted(MODEL_OUTPUT_KEYS - set(parsed.keys()))
        unexpected = sorted(set(parsed.keys()) - MODEL_OUTPUT_KEYS)
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected keys: {', '.join(unexpected)}")
        raise ModelOutputError("Invalid model response keys (" + "; ".join(details) + ").")

    article = parsed["article"]
    status = parsed["status"]
    explanation = parsed["explanation"]
    evidence = parsed["evidence"]

    if article is not None and (not isinstance(article, str) or article not in LABELS):
        raise ModelOutputError("Field 'article' must be a supported label or null.")
    if not isinstance(status, str) or status not in MODEL_STATUSES:
        raise ModelOutputError("Field 'status' must be 'classified' or 'ambiguous'.")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ModelOutputError("Field 'explanation' must be a non-empty string.")
    if not isinstance(evidence, list) or not all(
        isinstance(item, str) and item.strip() for item in evidence
    ):
        raise ModelOutputError("Field 'evidence' must be a list of non-empty strings.")
    if status == "classified" and article not in LABELS:
        raise ModelOutputError("Inconsistent output: classified status requires an article label.")
    if status == "ambiguous" and article is not None:
        raise ModelOutputError("Inconsistent output: ambiguous status requires article=null.")

    return {
        "article": article,
        "status": status,
        "explanation": explanation.strip(),
        "evidence": [item.strip() for item in evidence],
    }


def validate_model_answer(
    parsed: Mapping[str, Any],
    text: str,
    target: str,
) -> None:
    """Reject model answers that contradict the supplied target phrase."""
    observed_match = re.match(
        r"^(a|an|the)\b", target.strip(), flags=re.IGNORECASE
    )
    observed_article = (
        observed_match.group(1).casefold() if observed_match else None
    )

    article = parsed["article"]

    if article == "indefinite" and observed_article not in {"a", "an"}:
        raise ModelOutputError(
            "Unsupported indefinite label: the target does not begin with 'a' or 'an'."
        )

    if article == "definite" and observed_article != "the":
        raise ModelOutputError(
            "Unsupported definite label: the target does not begin with 'the'."
        )

    for item in parsed["evidence"]:
        # Reject evidence that explicitly names a different target phrase.
        target_match = re.search(
            r"\btarget(?: noun)? phrase\b[^'\"]*['\"]([^'\"]+)['\"]",
            item,
            flags=re.IGNORECASE,
        )
        if target_match:
            cited_target = target_match.group(1).strip()
            if cited_target.casefold() != target.strip().casefold():
                raise ModelOutputError(
                    "Evidence refers to a different target phrase."
                )

        
        # Validate only explicit positive claims naming an overt article.
        # Do not mistake phrases such as "no overt article" for positive claims.
        cited_article_match = re.search(
            r"\bovert article\s*(?::|=|is)\s*['\"]?(a|an|the)\b",
            item,
            flags=re.IGNORECASE,
        )

        if cited_article_match:
            cited_article = cited_article_match.group(1).casefold()

            if (
                observed_article is None
                or cited_article != observed_article
            ):
                raise ModelOutputError(
                    "Evidence claims an overt article not present in the target phrase."
                )


def build_messages(
    record: Mapping[str, Any],
    *,
    mode: str,
    retrieved_rules: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build baseline or agent chat messages, with consistent few-shot labels."""
    if mode not in {"baseline", "agent"}:
        raise ValueError("mode must be 'baseline' or 'agent'.")
    system_prompt = BASELINE_SYSTEM_PROMPT if mode == "baseline" else AGENT_SYSTEM_PROMPT
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for example_input, example_output in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": json.dumps(example_input, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(example_output, ensure_ascii=False)})

    current_input: dict[str, Any] = {
        "text": record["text"],
        "target_phrase": record["target_phrase"],
    }
    if mode == "agent":
        current_input["context"] = record.get("context") or ""
        current_input["retrieved_rules"] = list(retrieved_rules or [])
        current_input["instructions"] = (
            "Analyze the supplied target phrase in the supplied sentence. "
            "Use retrieved rules only when relevant to this example. "
            "First determine whether the target is a complete noun phrase "
            "within the scope of article analysis. "
            "If the target begins with 'a' or 'an', classify it as indefinite. "
            "If the target begins with 'the', classify it as definite. "
            "Never assign definite or indefinite unless the corresponding "
            "overt article begins the supplied target phrase. "
            "Do not infer zero article merely because 'a', 'an', or 'the' "
            "is absent. "
            "For a possible zero article, examine the noun type, syntactic "
            "construction, conventional usage, and supplied context. "
            "Distinguish bare plural nouns, mass nouns, proper names, "
            "and conventional institutional expressions. In 'go to school', "
            "'school' is singular and the zero-article use is institutional, "
            "not a bare plural. "
            "Treat proper names according to conventional article usage; "
            "some names take zero article and others include an article. "
            "An ordinary bare singular count noun may be ungrammatical "
            "in standard English and must not automatically be classified "
            "as zero article. "
            "Pronouns and other non-noun phrases are outside the intended "
            "scope and must return article=null and status='ambiguous'. "
            "If the target span is incomplete, an alternative determiner "
            "makes the case out of scope, or evidence is insufficient, "
            "return article=null and status='ambiguous'. "
            "Do not classify a phrase containing an alternative determiner "
            "as zero article merely because it lacks 'a', 'an', or 'the'. "
            "Every explanation and evidence item must match the actual "
            "sentence and target phrase. Do not invent context, identities, "
            "occupations, grammatical constructions, or rule citations. "
            "Do not misidentify a noun as a pronoun, a common noun as a "
            "proper name, or a mass noun as a bare plural. "
            "Return only the required JSON object."
        )

    messages.append({"role": "user", "content": json.dumps(current_input, ensure_ascii=False)})
    return messages


class OpenRouterClient:
    """Small OpenRouter transport wrapper with bounded retries and sanitized errors."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_url: str | None = None,
        timeout_seconds: float | None = None,
        *,
        max_attempts: int = MAX_API_ATTEMPTS,
        retry_delay_seconds: float = RETRY_DELAY_SECONDS,
    ) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

        if self.provider not in {"ollama", "openrouter"}:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'openrouter'.")

        # Select model and endpoint for the configured provider.
        if self.provider == "ollama":
            default_model = "llama3.2:3b"
            default_url = "http://localhost:11434/v1/chat/completions"
            provider_key = ""
        else:
            default_model = DEFAULT_MODEL
            default_url = DEFAULT_API_URL
            provider_key = os.getenv("OPENROUTER_API_KEY", "")

        self.model = (
            model
            or os.getenv("LLM_MODEL")
            or (
                os.getenv("OLLAMA_MODEL", default_model)
                if self.provider == "ollama"
                else os.getenv("OPENROUTER_MODEL", default_model)
            )
        )

        self.api_url = (
            api_url
            or os.getenv("LLM_BASE_URL")
            or (
                os.getenv("OLLAMA_BASE_URL", default_url)
                if self.provider == "ollama"
                else os.getenv("OPENROUTER_BASE_URL", default_url)
            )
        )

        self.api_key = (
            api_key if api_key is not None
            else provider_key
        )

        timeout_value = timeout_seconds
        if timeout_value is None:
            raw_timeout = os.getenv(
                "LLM_TIMEOUT_SECONDS",
                os.getenv(
                    "OPENROUTER_TIMEOUT_SECONDS",
                    str(DEFAULT_TIMEOUT_SECONDS),
                ),
            )
            try:
                timeout_value = float(raw_timeout)
            except ValueError as exc:
                raise ValueError("LLM_TIMEOUT_SECONDS must be a number.") from exc

        if timeout_value <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")

        self.timeout_seconds = timeout_value
        self.max_attempts = min(max_attempts, MAX_API_ATTEMPTS)
        self.retry_delay_seconds = retry_delay_seconds

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Call Chat Completions and return the first choice's message content.

        Retries only HTTP 429, HTTP 5xx, connection failures, and timeouts. It never
        retries other HTTP 4xx statuses, malformed API envelopes, or malformed model
        content. Authorization is passed only as a header and is never logged.
        """
        if self.provider == "openrouter" and not self.api_key:
            raise APIError(
                "missing_api_key",
                "OPENROUTER_API_KEY is not configured.",
            )

        headers = {
            "Content-Type": "application/json",
        }

        # OpenRouter requires a key. Local Ollama normally does not.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
        }

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=(5.0, self.timeout_seconds),
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                LOGGER.warning("OpenRouter transient transport failure on attempt %s/%s: %s", attempt, self.max_attempts, type(exc).__name__)
                if attempt < self.max_attempts:
                    time.sleep(self.retry_delay_seconds)
                    continue
                raise APIError(
                    "transport_retry_exhausted",
                    "OpenRouter connection or timeout failure after maximum attempts.",
                    retryable=True,
                ) from exc
            except requests.exceptions.RequestException as exc:
                # Other Requests errors (for example, an invalid URL) are not
                # known transient failures, but still become per-record errors.
                raise APIError(
                    "transport_error",
                    "OpenRouter request failed with a non-retryable transport error.",
                    retryable=False,
                ) from exc

            status_code = response.status_code
            is_retryable_status = status_code == 429 or 500 <= status_code <= 599
            if is_retryable_status:
                LOGGER.warning("OpenRouter retryable HTTP status %s on attempt %s/%s", status_code, attempt, self.max_attempts)
                if attempt < self.max_attempts:
                    time.sleep(self.retry_delay_seconds)
                    continue
                raise APIError(
                    "http_retry_exhausted",
                    f"OpenRouter returned retryable HTTP status {status_code} after maximum attempts.",
                    retryable=True,
                )
            if not 200 <= status_code < 300:
                # Deliberately omit response body and headers; they may contain details
                # unsuitable for logs or user-facing error records.
                raise APIError("http_error", f"OpenRouter returned HTTP status {status_code}.")

            try:
                body = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise APIError("invalid_api_json", "OpenRouter response body was not valid JSON.") from exc
            if not isinstance(body, dict):
                raise APIError("invalid_api_response", "OpenRouter response must be a JSON object.")
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise APIError("invalid_api_response", "OpenRouter response did not contain a valid choices list.")
            message = choices[0].get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise APIError("invalid_api_response", "OpenRouter response did not contain text message content.")
            return message["content"]

        # The loop always returns or raises. This guard keeps type checkers satisfied.
        raise APIError("unexpected_transport_state", "OpenRouter request ended unexpectedly.")


def classify_record(
    record: Mapping[str, Any],
    *,
    mode: str = "agent",
    knowledge_base: Mapping[str, Any] | None = None,
    client: OpenRouterClient | Any | None = None,
) -> dict[str, Any]:
    """Classify one input record and convert failures into structured results.

    ``client`` may be a small test double implementing ``complete(messages)``. An
    operational failure never becomes a linguistic ``ambiguous`` result and never
    receives a guessed article label.
    """
    if mode not in {"baseline", "agent"}:
        raise ValueError("mode must be 'baseline' or 'agent'.")

    validation_error = validate_record(record)
    if validation_error:
        return _make_result(
            record,
            article=None,
            status="error",
            explanation="Input validation failed; no linguistic classification was attempted.",
            evidence=[],
            error={"code": "invalid_input", "message": validation_error, "retryable": False},
        )

    text = record["text"].strip()
    target = record["target_phrase"].strip()
    # Preserve the exact input fields in every result; use a separate normalized
    # record for prompts and lexical checks.
    base_record = dict(record)
    base_record["context"] = record.get("context") or ""
    analysis_record = dict(base_record)
    analysis_record["text"] = text
    analysis_record["target_phrase"] = target

    if not target_phrase_occurs(text, target):
        return _make_result(
            base_record,
            article=None,
            status="ambiguous",
            explanation="The target phrase was not found as a contiguous span in the sentence, so its article usage cannot be grounded.",
            evidence=["The supplied target phrase does not occur in the supplied sentence."],
        )

    starts_with_overt_article = re.match(r"^(a|an|the)\b", target, flags=re.IGNORECASE) is not None
    if mode == "agent" and not starts_with_overt_article and target_span_has_preceding_determiner(text, target):
        return _make_result(
            base_record,
            article=None,
            status="ambiguous",
            explanation="The target may be an incomplete span inside a larger noun phrase because a determiner immediately precedes it.",
            evidence=["A determiner immediately precedes the supplied target span.", "The supplied target may not be the full noun phrase."],
            source="deterministic",
        )

    # Alternative determiners are outside the article labels supported
    # by this task. Abstain instead of incorrectly predicting zero article.
    alternative_determiner_pattern = re.compile(
        r"^(some|any|my|your|his|her|its|our|their|"
        r"this|that|these|those|each|every|either|neither|"
        r"no|another|many|much|few|several|all|both)\s+\S+",
        flags=re.IGNORECASE,
    )

    if mode == "agent" and alternative_determiner_pattern.match(target):
        return _make_result(
            base_record,
            article=None,
            status="ambiguous",
            explanation=(
                "The target phrase begins with a determiner other than "
                "'a', 'an', or 'the', which is outside the supported "
                "article labels for this task."
            ),
            evidence=[
                f"Target phrase: '{target}'",
                "The target begins with an alternative determiner.",
            ],
            source="deterministic",
        )
    
    if mode == "agent":

        direct_label = deterministic_article(text, target)
        if direct_label is not None:
            overt_article = re.match(
                r"^(a|an|the)\b", target, flags=re.IGNORECASE
            )
            assert overt_article is not None
            observed = overt_article.group(1).casefold()
            return _make_result(
                base_record,
                article=direct_label,
                status="classified",
                explanation=(
                    f"The verified target phrase begins with "
                    f"the overt article '{observed}'."
                ),
                evidence=[
                    f"Target phrase occurs in the sentence: '{target}'.",
                    f"Overt article: '{observed}'.",
                ],
                source="deterministic",
            )


    rules: list[dict[str, Any]] = []
    if mode == "agent":
        rules = knowledge_lookup(text, target, knowledge_base or {"rules": []})

    messages = build_messages(analysis_record, mode=mode, retrieved_rules=rules)
    active_client = client or OpenRouterClient()
    
    try:
        content = active_client.complete(messages)

        
        try:
            parsed = parse_model_output(content)

        except ModelOutputError as first_error:
            LOGGER.warning(
                "Model output parsing failed for record %r; retrying once.",
                base_record.get("id"),
            )

            repair_messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid: {first_error}. "
                        "Return exactly one valid JSON object with exactly "
                        "these keys: article, status, explanation, evidence. "
                        "Use article='definite', 'indefinite', or 'zero' "
                        "with status='classified', or article=null with "
                        "status='ambiguous'. Explanation must be a non-empty "
                        "string and evidence must be a list of non-empty "
                        "strings. Do not use Markdown fences or extra text. "
                        "Return only the corrected JSON object."
                    ),
                },
            ]

            content = active_client.complete(repair_messages)
            parsed = parse_model_output(content)

        # Validate the classification after JSON parsing and repair.
        validate_model_answer(parsed, text, target)


    except APIError as exc:
        LOGGER.error("Classification operation failed for record %r: %s", base_record.get("id"), exc.code)
        return _make_result(
            base_record,
            article=None,
            status="error",
            explanation="Classification could not be completed because of an API or transport failure.",
            evidence=[],
            source="fallback",
            error={"code": exc.code, "message": exc.message, "retryable": exc.retryable},
            fallback_used=True,
        )
    except ModelOutputError as exc:
        LOGGER.error("Invalid model output for record %r: %s", base_record.get("id"), str(exc))
        return _make_result(
            base_record,
            article=None,
            status="error",
            explanation="The model response did not satisfy the required output schema; no label was accepted.",
            evidence=[],
            source="fallback",
            error={"code": "invalid_model_output", "message": str(exc), "retryable": False},
            fallback_used=True,
        )


    return _make_result(
        base_record,
        article=parsed["article"],
        status=parsed["status"],
        explanation=parsed["explanation"],
        evidence=parsed["evidence"],
        source="llm",
    )


def process_records(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str = "agent",
    knowledge_base: Mapping[str, Any] | None = None,
    client: OpenRouterClient | Any | None = None,
) -> list[dict[str, Any]]:
    """Process records independently, preserving order and rejecting duplicate IDs."""
    seen_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    for record in records:
        record_id = record.get("id")
        if isinstance(record_id, str) and record_id.strip():
            normalized_id = record_id.strip()
            if normalized_id in seen_ids:
                results.append(
                    _make_result(
                        record,
                        article=None,
                        status="error",
                        explanation="Input validation failed; duplicate identifiers are not allowed in a batch.",
                        error={"code": "duplicate_id", "message": f"Duplicate id: {record_id}", "retryable": False},
                    )
                )
                continue
            seen_ids.add(normalized_id)
        results.append(classify_record(record, mode=mode, knowledge_base=knowledge_base, client=client))
    return results


def read_csv_records(path: str | Path) -> list[dict[str, Any]]:
    """Read input records from CSV using the standard library."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is empty or has no header row.")
        return [dict(row) for row in reader]


def write_jsonl(results: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write one UTF-8 JSON result per line, preserving result order."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(dict(result), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read prediction JSONL and report line-specific malformed JSON errors."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}.") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object.")
            records.append(item)
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify article usage for specified English noun phrases.")
    parser.add_argument("--input", default="input.csv", help="Input CSV path (default: input.csv).")
    parser.add_argument("--output", default="outputs/predictions.jsonl", help="Output JSONL path.")
    parser.add_argument("--mode", choices=("baseline", "agent"), default="agent", help="Classification configuration.")
    parser.add_argument("--knowledge-base", default="knowledge_base.json", help="Local rules JSON path.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        records = read_csv_records(args.input)
        knowledge_base = load_knowledge_base(args.knowledge_base) if args.mode == "agent" else None
        results = process_records(records, mode=args.mode, knowledge_base=knowledge_base)
        write_jsonl(results, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("Could not run classification: %s", exc)
        return 2

    error_count = sum(item["status"] == "error" for item in results)
    ambiguous_count = sum(item["status"] == "ambiguous" for item in results)
    LOGGER.info("Wrote %s records to %s (%s ambiguous, %s errors).", len(results), args.output, ambiguous_count, error_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
