# Insurance Claim Processing Agent (LangGraph)

An AI-powered insurance claims workflow built with **LangGraph**, wrapped in a **Streamlit** UI. It verifies documents, checks policy eligibility, screens for fraud, summarizes the claim, and routes each case to auto-approval, auto-rejection, or a human reviewer.

## Architecture

```
                          START
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
   Document Verification  Eligibility   Fraud Detection      <- run in PARALLEL
             └──────────────┼──────────────┘
                            ▼
                          merge
                            ▼
                    Claim Summary Agent
                            ▼
                     Decision Router  (conditional edge)
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
      auto_approve     auto_reject   Human Approval Agent
             │              │         (interrupt() - HITL)
             ▼              ▼              ▼
            END            END            END
```

### The 5 required agents/nodes

| # | Agent | File | What it does |
|---|-------|------|---------------|
| 1 | **Document Verification Agent** | `nodes.py::document_verification_node` | Compares submitted documents against the required list for the claim type; flags what's missing. |
| 2 | **Eligibility Check Agent** | `nodes.py::eligibility_check_node` | Confirms the incident date falls within the policy period and the claim type is covered. |
| 3 | **Fraud Detection Agent** | `nodes.py::fraud_detection_node` | Scores fraud risk 0–100 from signals like claim/coverage ratio, how recently the policy started, round-number amounts, and claim history. |
| 4 | **Claim Summary Agent** | `nodes.py::claim_summary_node` | Uses the LLM to write a short underwriter-facing summary combining the three checks above. |
| 5 | **Human Approval Agent** | `nodes.py::human_approval_node` | Pauses the graph with LangGraph's `interrupt()` when a claim is escalated, and resumes once a reviewer submits a decision in the Streamlit UI. |

Two small support nodes (`merge_node`, `decision_node`) handle the parallel fan-in and the rule-based routing logic, respectively - this is what LangGraph's **conditional edges** dispatch on.

### Parallel execution
`document_verification`, `eligibility_check`, and `fraud_detection` all have an edge directly from `START`, so LangGraph schedules them concurrently. They all edge into a single `merge` node, which only fires once all three branches finish (a standard LangGraph fan-out/fan-in pattern). The `trace` field in state uses an `operator.add` reducer so the three branches can each append to the execution log without conflicting.

### Human-in-the-loop
`human_approval_node` calls `langgraph.types.interrupt(...)`, which pauses graph execution using the `MemorySaver` checkpointer and returns the payload to the caller. The Streamlit app detects `"__interrupt__"` in the result, renders an Approve/Reject form, and resumes the same thread with `graph.invoke(Command(resume={...}), config)`.

### Routing rules (`nodes.py::decision_node`)
- Missing documents → **reject**
- Not eligible (expired policy / uncovered claim type) → **reject**
- Fraud score ≥ 90 → **reject** (very high confidence fraud)
- Fraud score ≥ 60, OR claim amount ≥ $15,000 → **human_review**
- Otherwise → **auto_approve**

Thresholds live in `config.py` and are easy to tune.

## LLM usage
Every node calls `llm.py::call_llm(system_prompt, user_prompt)`. The **sidebar** in the Streamlit app lets you pick the provider at runtime - no code edits, no files to touch:

- **Mock (no key needed)** - deterministic rule-based reasoning, so the whole graph and all 5 demo scenarios run end-to-end for free. This is the default.
- **DeepSeek** - paste your own DeepSeek API key into the password-masked field. DeepSeek exposes an OpenAI-compatible endpoint, so this reuses `langchain_openai.ChatOpenAI` pointed at `https://api.deepseek.com` with model `deepseek-chat` (editable in the sidebar).
- **OpenAI** - paste your own OpenAI key, model defaults to `gpt-4o-mini` (editable).

The key you paste is **never written to disk or into any file** - it lives only in that browser session's memory (`st.session_state` → passed straight into the API client for that run). Refreshing the page or closing the tab clears it.

If a call fails (bad key, insufficient quota, network issue), the app shows a clear inline error instead of crashing or silently falling back - so you always know whether you're looking at live-model output or mock output.

You can also set a key via environment variable instead of the UI, for non-Streamlit use (scripts, tests):
```bash
export DEEPSEEK_API_KEY=sk-...
# or
export OPENAI_API_KEY=sk-...
```

## Setup

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Files
- `state.py` - shared `ClaimState` TypedDict (with the parallel-safe `trace` reducer)
- `config.py` - required documents per claim type + fraud/value thresholds
- `llm.py` - LLM wrapper (real OpenAI call or mock fallback)
- `nodes.py` - the 5 agents + decision router + merge/terminal nodes
- `graph.py` - `StateGraph` wiring: parallel fan-out, fan-in, conditional edges, checkpointer
- `streamlit_app.py` - UI: scenario picker, custom claim builder, results dashboard, human review form

## Test scenarios (all verified to produce the expected routing)

| # | Scenario | Key signal | Result |
|---|----------|-----------|--------|
| 1 | Complete claim, valid documents | All docs present, policy active, low fraud score | **Auto Approve** |
| 2 | Missing required documents | Property claim missing repair estimate + ownership proof | **Reject** |
| 3 | Suspicious claim amount | Claim is 94% of coverage limit + policy bought 12 days before incident → fraud score 85 | **Human Review** |
| 4 | Expired policy | Incident date after `policy_end_date` | **Reject** |
| 5 | High-value claim, valid documents | $32,000 claim ≥ $15,000 high-value threshold, otherwise clean | **Human Review** |

All five are selectable from the sidebar dropdown in the Streamlit app ("Preloaded demo scenario" mode), or you can build a fully custom claim ("Custom claim" mode) to test other combinations.

## Notes / possible extensions
- Swap `config.py`'s hard-coded rules for a real policy database lookup.
- Persist the `MemorySaver` checkpointer to a database (e.g. `SqliteSaver`/`PostgresSaver`) so pending human-review claims survive an app restart.
- Add a real document-upload step (PDF/image) and have the Document Verification Agent do OCR-based extraction instead of a checklist match.
