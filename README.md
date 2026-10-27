# English Article Analysis Agent

A small, runnable portfolio MVP that classifies the English article usage of one specified target noun phrase in a sentence. It demonstrates requirement analysis, a typed input/output contract, deterministic routing, local knowledge retrieval, OpenRouter integration, structured-output validation, bounded retries, abstention, testing, and offline evaluation tooling.

> **Evaluation honesty:** the repository's tests are run with mocks and do not call OpenRouter. No live model-quality score is claimed or prepopulated. Run the evaluation command with your own API key to write actual baseline and agent metrics to `outputs/evaluation/metrics.json`.

## Problem and intended user

The tool is intended as a linguistic-analysis prototype and an entry-level AI Agent Engineer portfolio project. One input record names a sentence and the target noun phrase to be classified. The system must classify the target phrase—not the sentence as a whole—as `definite`, `indefinite`, or `zero`. It also needs a principled way to say that a case is unresolved.

Example:

| Sentence | Target phrase | Expected article |
|---|---|---|
| `I saw a dog near the house.` | `a dog` | `indefinite` |
| `I saw a dog near the house.` | `the house` | `definite` |
| `Dogs are great.` | `Dogs` | `zero` |
| `I bought an umbrella.` | `an umbrella` | `indefinite` |

The same sentence can therefore produce different labels for different target phrases.

## Linguistic scope and assumptions

- **Definite:** the target noun phrase uses overt `the` as its article. Proper names with conventional `the` (for example, `the Netherlands`) still receive `definite`.
- **Indefinite:** the target noun phrase uses overt `a` or `an` as its article. The model sees the full phrase and should distinguish `a` from `an` by form, not by spelling alone.
- **Zero:** a well-formed noun phrase has no overt `a/an/the`, no competing determiner, and is a licensed bare noun phrase. Typical cases include generic bare plurals (`Dogs are great`), some specific bare plurals (`I spoke to neighbors from my building`), bare mass nouns (`Music helps me concentrate`), ordinary proper names (`Paris`), and some conventional institutional expressions (`go to school`, `go to bed`, `at work`).
- **Not automatically zero:** a missing target span; a pronoun; a possessive, demonstrative, or quantified NP such as `my brother`, `this house`, or `some information`; or a bare singular count noun in a possibly malformed sentence (`I bought book yesterday`). These are `ambiguous`/unresolved in this MVP unless the target can be interpreted reliably.
- **Proper nouns:** many ordinary names have zero article, but some place names conventionally contain `the`. Lexical conventions are not exhaustively encoded.
- **Institutional expressions:** meaning and dialect can matter (`go to school`, `at school`, `in hospital`, `the hospital`). The small rule base offers guidance but is not a complete dialect atlas.
- **Generic vs specific reference:** reference is not identical to article form. A bare plural can denote a kind or a contextually specific group and still have zero article. An overt `the` remains definite even when the surrounding context is sparse.
- **Annotation uncertainty:** the gold set contains judgments that reasonable annotators may debate, especially around institutional meanings, elliptical mass-noun phrases, dialect, and incomplete target spans. The dataset is representative, not exhaustive, and should be reviewed by an English linguist before being used as a benchmark.

`status` in prediction output is separate from `article`:

- `classified`: article is one of the three labels.
- `ambiguous`: `article` is JSON `null`; the system abstains for linguistic/scope/span reasons.
- `error`: `article` is JSON `null`; input validation, API, transport, or model-schema handling failed. This is an operational result, not linguistic ambiguity.

## Architecture

1. Validate `id`, `text`, `target_phrase`, and optional `context`; reject blank required fields and duplicate batch IDs.
2. In agent mode, resolve a narrow set of high-confidence cases deterministically: a verified target phrase beginning with overt `a`, `an`, or `the`. Also abstain when the target does not occur or a determiner immediately precedes a likely incomplete target span.
3. Retrieve relevant local rules from `knowledge_base.json` through `knowledge_lookup(text, target_phrase, knowledge_base)`. Retrieval is keyword-based and deterministic; it returns guidance and examples, not a prediction.
4. Send harder cases to OpenRouter's Chat Completions API using a few-shot prompt. Agent mode includes context and retrieved rules; baseline mode uses a concise prompt without the knowledge tool/context.
5. Parse model text as JSON and validate exact keys, types, supported labels, non-empty explanation/evidence, and consistency between `article` and `status`.
6. Retry retryable HTTP/transport failures no more than three times, wait two seconds between attempts, and emit a structured `error` result after exhausted attempts. Do not guess a label on an operational failure.
7. Write one JSON object per line and evaluate baseline/agent on the same held-out set with coverage and operational rates alongside accuracy.

This is an **LLM application with a tool-using fixed pipeline**, not an autonomous agent. Python selects the route and calls local retrieval. Native provider tool calling is not required.

## Project files

```text
english-article-analysis-agent/
├── main.py              # validation, strategy selection, knowledge tool, API client, JSONL I/O
├── eval.py              # held-out metrics and baseline-vs-agent comparison
├── test_main.py         # pytest unit/integration tests with mocks only
├── check_openrouter_connection.py # optional live OpenRouter connectivity check
├── knowledge_base.json  # curated linguistic rules and examples
├── input.csv            # small demonstration batch
├── golden.csv           # dev/test annotations plus excluded robustness fixtures
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── DESIGN.md
```

`golden.csv` uses `split=dev` for prompt/rule iteration and `split=test` for held-out comparison. Four `split=robustness` rows represent malformed-input, retryable API-failure, and invalid-model-output fixtures; they are excluded from linguistic accuracy denominators because an API outage is not an article label.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
```

For OpenRouter, put your own key in `.env` as `OPENROUTER_API_KEY`. Do not commit `.env`. The default provider is OpenRouter; the model defaults to `meta-llama/llama-3-8b-instruct:free` and can be changed with `OPENROUTER_MODEL`. The API endpoint defaults to `https://openrouter.ai/api/v1/chat/completions`; `OPENROUTER_BASE_URL` can override it. `OPENROUTER_TIMEOUT_SECONDS` controls the read timeout; a 5-second connect timeout is also set in the client.

For local Ollama, install and start Ollama, download a chat model (for example, `ollama pull llama3.2:3b`), and set these values in your local `.env` instead:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434/v1/chat/completions
LLM_TIMEOUT_SECONDS=120
```

Ollama runs locally and does not require an OpenRouter API key. Keep `.env` private for either setup.

## Usage

Classify the demo CSV in agent mode:

```bash
python main.py --input input.csv --output outputs/predictions.jsonl --mode agent
```

Run the LLM-only baseline:

```bash
python main.py --input input.csv --output outputs/baseline.jsonl --mode baseline
```

Agent mode can classify overt-article cases without an API call. Bare noun phrases that require interpretation need a configured API key. Each result includes the original `id`, `text`, and `target_phrase`, as well as `article`, `status`, `explanation`, `evidence`, `source`, `error`, and `fallback_used`.

Example JSONL result for `a dog`:

```json
{"id":"I01","text":"I saw a dog near the house.","target_phrase":"a dog","article":"indefinite","status":"classified","explanation":"The verified target phrase begins with the overt article 'a'.","evidence":["Target phrase occurs in the sentence: 'a dog'.","Overt article: 'a'."],"source":"deterministic","error":null,"fallback_used":false}
```

The displayed result illustrates the deterministic output contract; it is not a live API response.

## Tests

```bash
pytest -q
```

Tests use `unittest.mock`/pytest monkeypatching and small test doubles. No test requires a key or network. They cover input validation, target-specific classification, ambiguity, strict JSON schema validation, 429/5xx/connection/timeout retries, retry limits and sleep behavior, non-retryable 4xx handling, knowledge lookup, JSONL order/IDs, missing predictions, operational failures, and an end-to-end mocked batch. The optional `check_openrouter_connection.py` script makes a live OpenRouter request and is intentionally not named as a pytest test; run it only when you want to verify your OpenRouter key and connection.

## Evaluation

Run the held-out test split with both configurations:

```bash
python eval.py --golden golden.csv --split test --output-dir outputs/evaluation
```

To iterate on prompts/rules without reporting it as final performance, use `--split dev`.

The evaluation writes:

- `baseline_predictions.jsonl`
- `agent_predictions.jsonl`
- `metrics.json`

Recompute metrics from saved results without making new API calls:

```bash
python eval.py \
  --golden golden.csv \
  --split test \
  --baseline-predictions outputs/evaluation/baseline_predictions.jsonl \
  --agent-predictions outputs/evaluation/agent_predictions.jsonl \
  --output-dir outputs/evaluation-recomputed
```

### Metric definitions and denominators

- **Accuracy on classifiable examples:** correct classified labels / all gold rows with `expected_status=classified` and a supported gold label. An abstention, operational error, or missing prediction counts as incorrect in this denominator.
- **Per-class precision, recall, F1:** calculated on those same classifiable gold rows. An abstention for a classifiable example is a false negative for its gold class.
- **Macro-F1:** unweighted mean of defined per-class F1 scores.
- **Coverage:** confidently classified predictions / all linguistic rows in the selected split, including gold-ambiguous cases.
- **Ambiguity rate:** predictions with `status=ambiguous` / all linguistic rows.
- **Fallback rate:** rows with `fallback_used=true` or `source=fallback` / all linguistic rows. API/schema failures produce a structured fallback record with `article=null`; they do not invent a label.
- **Operational-failure rate:** `status=error` plus missing predictions / all linguistic rows. It is distinct from ambiguity; missing result IDs are explicitly counted as operational failures.
- **Ambiguity recall:** ambiguous predictions / gold rows annotated as ambiguous.
- **Performance by phenomenon:** category-specific accuracy uses classifiable gold rows in that category; category coverage uses every linguistic row in that category.

The report records denominators and counts so a high score obtained by abstaining on difficult cases is visible. It also includes `example_diagnostics` (one outcome per gold ID) and an `error_analysis` summary with failure counts by phenomenon and data-driven next-step suggestions. The `golden.csv` robustness fixtures are not sent to the model during normal language evaluation: API failures are simulated in tests instead.

### Actual results

No actual baseline-vs-agent model metrics are bundled. The build/test process did not make a live OpenRouter call, so a real accuracy comparison has not been verified here. The command above records actual results in `metrics.json` when run in an environment with API access. Do not interpret mocked unit-test results as model quality.

## Design trade-offs

- A small Python orchestration layer is easier to inspect and test than introducing LangChain/LangGraph, an MCP server, a vector database, or a database before there is evidence that those components improve the task.
- The local knowledge tool uses transparent keyword retrieval. It is intentionally simple and can be upgraded only if error analysis shows rule selection is a bottleneck.
- The deterministic fast path is narrow. It uses only observed article form and target-span checks; it does not pretend to parse full syntax.
- The baseline is intentionally simpler, while the agent adds context, curated guidance, deterministic high-confidence checks, and explicit abstention. Output schema validation and safe result formatting remain common to both configurations.

## Limitations and future improvements

The MVP does not implement a full parser, coreference resolver, dialect-specific grammar, or expert adjudication. Keyword rules are not exhaustive. LLM explanations are model-generated and are never treated as proof of correctness. Future improvements should start from adjudicated errors: add targeted gold cases, measure changed behavior on the dev set, then freeze and report on the held-out test set. Further options include a structured annotation UI, inter-annotator agreement, calibration/abstention thresholds, a small API wrapper, and an MCP adapter if a real client workflow needs it.
