# AI Resume Screening Assistant (LangChain + RAG)

A Streamlit app that lets a recruiter upload resumes (PDF), paste a Job
Description, and get a grounded, structured evaluation of each candidate:
**Match Score, Matching/Missing Skills, Summary, Strengths, Weaknesses, and
a Hiring Recommendation** - generated only from what's actually retrieved
out of each resume via RAG.

## Architecture

```
Resume PDF ──► PyPDFLoader ──► RecursiveCharacterTextSplitter ──► chunks
                                                                     │
                                                FastEmbedEmbeddings (local, free)
                                                                     │
                                                                     ▼
                                                          FAISS vector store
                                                             (per resume)
                                                                     │
Job Description ──► multi-query retrieval (JD + skills/exp/education) ──► retriever
                                                                     │
                                                                     ▼
                                ChatPromptTemplate + ChatOpenAI (DeepSeek or OpenAI)
                                .with_structured_output(CandidateEvaluation)
                                                                     │
                                                                     ▼
                                        Pydantic-validated JSON evaluation
                                                                     │
                                                                     ▼
                                              Streamlit UI (cards, compare, rank)
```

**LLM provider is pluggable** - pick **DeepSeek** (cheap/low-cost, OpenAI-compatible
API) or **OpenAI** from the sidebar. **Embeddings always run locally for free**
via FastEmbed (ONNX, CPU, no API key, no per-resume cost) regardless of which
LLM provider you choose - this matters because DeepSeek doesn't expose an
embeddings endpoint at all.

| Requirement | Implementation |
|---|---|
| LangChain | `langchain`, `langchain-core`, `langchain-community`, `langchain-openai` |
| LLM | `ChatOpenAI` pointed at either `https://api.deepseek.com` (`deepseek-chat`) or the default OpenAI endpoint (`gpt-4o-mini`), switchable in the sidebar |
| PDF Document Loader | `PyPDFLoader` |
| Text Splitter | `RecursiveCharacterTextSplitter` |
| Embedding Model | `FastEmbedEmbeddings` (`BAAI/bge-small-en-v1.5`) - local, free, no API key |
| Vector Database | `FAISS` (one in-memory index per resume) |
| Retriever | `vectorstore.as_retriever()`, multi-query (JD + skills/exp/education sub-queries) |
| Prompt Template | `ChatPromptTemplate` (system + human messages) |
| Output Parser | Pydantic `CandidateEvaluation` model via `.with_structured_output(..., method="function_calling")` |
| Streamlit app | `app.py` |

> `method="function_calling"` is used instead of the newer strict JSON-schema
> mode because that stricter mode is OpenAI-only - function calling is
> supported by both OpenAI and DeepSeek, so evaluations work either way.

Each resume gets **its own vector store**, so evaluations are strictly
grounded in that candidate's own content - nothing is cross-contaminated
between candidates, and the model is explicitly instructed not to invent
skills or experience that aren't in the retrieved context.

## Project Structure

```
resume_screener/
├── app.py                          # Streamlit UI
├── rag_engine.py                   # RAG pipeline, schema, prompt
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml                 # Custom violet theme (consistent look regardless of system dark/light mode)
├── assets/
│   ├── background.png              # Active gradient background image
│   └── background_alt.png          # Alternate gradient background (see "Design" below)
└── sample_data/
    ├── jd_data_scientist.txt       # Sample JD for testing
    ├── generate_sample_resumes.py  # Regenerates the sample PDFs below
    ├── Resume_A.pdf                # Strong match (Data Scientist)
    ├── Resume_B.pdf                # Moderate match (Backend SWE)
    └── Resume_C.pdf                # Weak match (Business Analyst intern)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), then in the
sidebar pick a **Provider** (DeepSeek or OpenAI) and paste the matching API
key. Get a DeepSeek key at https://platform.deepseek.com/api_keys - no
OpenAI account is required.

(Optional: instead of pasting the key each time, set `DEEPSEEK_API_KEY` or
`OPENAI_API_KEY` as an environment variable before launching, and the sidebar
will pre-fill it.)

The first run will download the local embedding model (~130MB, one-time,
requires normal internet access) - after that it's cached and runs offline.

## Using the Sample Data

1. In the sidebar, pick a provider and enter your API key.
2. Copy the contents of `sample_data/jd_data_scientist.txt` into the **Job
   Description** box.
3. Upload `Resume_A.pdf`, `Resume_B.pdf`, and `Resume_C.pdf` from
   `sample_data/`.
4. Click **Evaluate resumes against JD**.

(To regenerate the sample PDFs: `python sample_data/generate_sample_resumes.py`.)

## Example Test Cases

All five example scenarios from the brief are covered by the app's tabs:

1. **Evaluate Resume A for a Data Scientist role**
   → Load the JD + `Resume_A.pdf` only, click Evaluate, open the
   **Individual Evaluations** tab. Expect a high match score - Resume A
   covers Python/scikit-learn/TensorFlow/PyTorch, AWS, MLflow/Airflow, A/B
   testing, and an M.S. in Computer Science.

2. **Compare Resume A and Resume B for the same JD**
   → Upload both, evaluate, go to the **Compare Two** tab, pick Resume A vs
   Resume B. Resume A (data scientist) should outscore Resume B (backend
   engineer with only light pandas/AWS exposure and no ML/stats background).

3. **Identify missing skills in Resume C**
   → Evaluate `Resume_C.pdf`, open its card in **Individual Evaluations**
   and read the **Missing Skills** field. Expect gaps like Python, machine
   learning frameworks (TensorFlow/PyTorch), SQL depth, A/B testing/statistics,
   and cloud platforms - Resume C is an Excel/PowerBI business-analyst intern.

4. **Recommend the best candidate among multiple resumes**
   → Upload all three, evaluate, open the **Rank & Recommend** tab. Candidates
   are sorted by match score (ties broken by recommendation strength), with
   the top candidate highlighted along with the model's justification.

5. **Generate a hiring recommendation with justification**
   → Every card in **Individual Evaluations** shows a `recommendation`
   (`Strong Hire` / `Hire` / `Maybe` / `No Hire`) plus a `justification`
   string citing specific resume evidence.

## Design

The UI uses a full-page gradient background image (`assets/background.png`)
with a frosted-glass ("glassmorphism") look layered on top for readability:
- The hero header, candidate cards, and sidebar all use a semi-transparent
  white background + backdrop blur, so text stays crisp over the busy image
- A light white gradient overlay is baked into the background CSS so the
  image never fights with text contrast
- Circular score "rings" color-coded by band (green ≥80, blue 65-79, amber 45-64, red <45)
- Pill-shaped skill chips (green = matching, red = missing)
- Colored recommendation badges (🌟 Strong Hire, 👍 Hire, 🤔 Maybe, 🚫 No Hire)
- Card-based layout with hover lift, used consistently across all four tabs
- Medal icons (🥇🥈🥉) + horizontal score bars in the ranking leaderboard

**To change the background image**: replace `assets/background.png` with any
image of the same name (a second option, `assets/background_alt.png`, is
included - just rename it to `background.png` to switch). If the file is
ever missing, `app.py` automatically falls back to a CSS-only gradient so the
app never breaks.

## Notes / Design Decisions

- **Multi-query retrieval**: instead of a single similarity search against
  the raw JD, the retriever runs against the JD text *and* three targeted
  sub-queries (skills, experience, education) so the LLM sees a well-rounded
  slice of the resume rather than only the single most JD-similar paragraph.
- **Grounding**: the system prompt explicitly forbids inventing skills or
  experience not present in the retrieved context, and untouched information
  is treated as absent rather than assumed.
- **Structured output**: `ChatOpenAI.with_structured_output(CandidateEvaluation)`
  guarantees a schema-valid Pydantic object every time (score bounds,
  enum-constrained recommendation, typed lists) - no manual JSON parsing.
- **Isolation**: each resume gets its own FAISS index, so retrieval for
  Candidate A can never leak Candidate B's content into the same evaluation.
- **Swap-able models**: `gpt-4o-mini` is the default per the brief, but the
  sidebar lets you switch to `gpt-4o` / `gpt-4.1-mini`; you could similarly
  swap `OpenAIEmbeddings`/`FAISS` for `HuggingFaceEmbeddings`/`Chroma` inside
  `rag_engine.py` without touching `app.py`.
