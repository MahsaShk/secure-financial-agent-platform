# secure-financial-agent-platform

A small local agent that answers questions from financial report text, runs
arithmetic through a fixed set of calculator operations, and refuses common
prompt-injection attempts. It is an MVP for testing this approach, not
financial advice.

The data in this repo is synthetic and only for smoke tests.

## Quick start

Requires Python 3.10 or newer. No API key is needed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

Run the smoke evaluation:

```powershell
python scripts\run_eval.py
```

Start the demo:

```powershell
python -m pip install -e ".[demo]"
streamlit run app.py
```

## How it works

```text
Question -> Input guardrail -> TF-IDF retrieval -> Retrieved-content guardrail
                                                -> Planner (rule-based)
                                                   |- arithmetic -> Policy check -> Calculator
                                                   |- no arithmetic -> Extractive answer
                                                -> AgentResponse + audit event
```

- **Retrieval** uses TF-IDF  with cosine similarity.
- **Planner** decides whether a question needs arithmetic and which operation
  to use. It then finds the operands in the retrieved text with regex. Choosing
  the operation and finding the operands are separate functions, so the choice
  can later be replaced with a model.
- **Calculator** accepts `subtract`, `percent_change`, and `percentage`,
  with finite numbers.
- **Guardrails** check both the question and the retrieved text for common
  injection phrases. The regex list is a baseline, not complete
  protection.
- **Responses** include the answer, citations, evidence excerpts, confidence,
  and a trace ID. Refusals include a reason code. Each request writes an audit
  event with the decision, retrieved chunk IDs, tool calls, and versions.

## Evaluation

`data/sample_eval.json` is a synthetic fixture. The evaluator
compares three setups: no retrieval, retrieval-only, and the full agent. It
reports answer accuracy, Recall@k, MRR, citation precision, tool-selection
accuracy, refusal accuracy, and p50/p95 latency.

`data/security_cases.json` has 20 attack and benign-utility cases. They show
that the pipeline runs. They are not a security benchmark.

These are smoke-test results on synthetic data. They are not FinQA results.

## Limitations

- Regex guardrails miss novel attacks and can flag benign text. For example,
  "act as the developer relations contact" trips `role_manipulation`.
- TF-IDF misses synonyms. "Sales" does not match "revenue."
- The planner recognizes just the phrasings in `planner.py`: growth amount,
  percent change, percentage of, and operating margin.
- Questions with more than one part are not supported. The lookup part is
  dropped, and the answer covers only the arithmetic.
- Operand extraction is regex over sentences. Metric aliases such as "assets"
  can match more than intended.
- Citations name the top-ranked chunk's source, even when several chunks
  were used.
- `Evidence.excerpt` is cut at 280 characters.
- `retrieval_recall_at_k` counts any relevant chunk, which overstates recall
  when evidence spans several chunks.
- The audit log is a local JSONL file, not an immutable store. The Streamlit
  app uses the in-memory sink.
- There is no authentication, authorization, or hosted API.
- Accuracy on FinQA has not been measured.

## Data and licenses

The committed fixtures are synthetic. FinQA is available from its
[project page](https://finqasite.github.io/) and
[repository](https://github.com/czyssrs/FinQA). 

Project code is licensed under the MIT License.
