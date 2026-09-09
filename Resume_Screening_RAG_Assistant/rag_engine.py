"""
rag_engine.py
--------------
Core RAG (Retrieval-Augmented Generation) engine for the AI Resume Screening Assistant.

Pipeline for each resume:
    PDF  --(PyPDFLoader)-->  raw text
         --(RecursiveCharacterTextSplitter)-->  chunks
         --(FastEmbedEmbeddings, local & free)-->  vectors
         --(FAISS)-->  vector store
         --(retriever)-->  JD-relevant chunks
         --(ChatPromptTemplate + structured LLM)-->  CandidateEvaluation (Pydantic)

Everything the LLM says about a candidate is grounded ONLY in chunks retrieved
from that candidate's resume + the job description supplied by the recruiter.

LLM provider is pluggable: OpenAI or any OpenAI-compatible endpoint (e.g.
DeepSeek). Embeddings always run locally via FastEmbed (ONNX, CPU, free, no
API key) so indexing never depends on - or costs money against - the chat
provider's quota.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# --------------------------------------------------------------------------- #
# 0. Supported LLM providers (all speak the OpenAI-compatible chat API)
# --------------------------------------------------------------------------- #
PROVIDERS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "OpenAI": {
        "base_url": None,  # use the SDK default
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    },
}

# Cached embedder - loading the ONNX model is the slow part, so every
# ResumeRAGEngine instance in the session reuses the same one.
_EMBEDDER: Optional[FastEmbedEmbeddings] = None


def get_embedder() -> FastEmbedEmbeddings:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _EMBEDDER


# --------------------------------------------------------------------------- #
# 1. Structured output schema (the "Output Parser" requirement)
# --------------------------------------------------------------------------- #
class CandidateEvaluation(BaseModel):
    """Structured evaluation of a single candidate against a Job Description."""

    candidate_name: str = Field(
        description="Candidate's name as found in the resume. Use the filename "
        "(without extension) if no name can be identified."
    )
    match_score: int = Field(
        description="Overall match score between 0 and 100, where 100 is a "
        "perfect match to the job description.",
        ge=0,
        le=100,
    )
    matching_skills: List[str] = Field(
        description="Skills/technologies/qualifications required by the JD that "
        "ARE evidenced in the resume."
    )
    missing_skills: List[str] = Field(
        description="Skills/technologies/qualifications required by the JD that "
        "are NOT evidenced in the resume."
    )
    summary: str = Field(
        description="A concise 2-4 sentence summary of the candidate's background "
        "as it relates to the JD."
    )
    strengths: List[str] = Field(
        description="Concrete strengths of this candidate for this specific role."
    )
    weaknesses: List[str] = Field(
        description="Concrete weaknesses, gaps, or risk areas for this specific role."
    )
    recommendation: Literal["Strong Hire", "Hire", "Maybe", "No Hire"] = Field(
        description="Overall hiring recommendation."
    )
    justification: str = Field(
        description="1-3 sentence justification for the recommendation, referencing "
        "specific evidence from the resume."
    )


# --------------------------------------------------------------------------- #
# 2. Prompt template
# --------------------------------------------------------------------------- #
EVALUATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter and resume screening assistant. "
            "You evaluate ONLY based on the resume excerpts provided to you as context "
            "- never invent experience, skills, or credentials that are not present in "
            "the context. If information is not present in the context, treat it as "
            "absent/unknown rather than assuming it exists.\n\n"
            "You will be given:\n"
            "1. A Job Description (JD)\n"
            "2. Retrieved excerpts from ONE candidate's resume (via RAG)\n\n"
            "Produce a rigorous, evidence-based evaluation of this candidate against "
            "the JD, following the required output schema exactly.",
        ),
        (
            "human",
            "JOB DESCRIPTION:\n{jd}\n\n"
            "RETRIEVED RESUME EXCERPTS (candidate: {source_name}):\n{context}\n\n"
            "Evaluate this candidate against the job description above.",
        ),
    ]
)

# Sub-queries used to pull a well-rounded set of chunks out of the resume,
# rather than relying on a single similarity search against the raw JD text.
RETRIEVAL_QUERIES_TEMPLATE = [
    "{jd}",
    "technical skills, tools, technologies, and programming languages",
    "work experience, job titles, responsibilities, and achievements",
    "education, degrees, certifications, and qualifications",
]


class ResumeRAGEngine:
    """
    One instance manages the RAG pipeline for a single uploaded resume:
    load -> split -> embed -> index -> retrieve -> evaluate.
    """

    def __init__(
        self,
        pdf_path: str,
        source_name: str,
        api_key: str,
        llm_model: str = "deepseek-chat",
        base_url: Optional[str] = "https://api.deepseek.com",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        retriever_k: int = 4,
    ):
        self.pdf_path = pdf_path
        self.source_name = source_name
        self.retriever_k = retriever_k

        # Embeddings are always local/free (FastEmbed, ONNX, CPU) - this
        # keeps indexing working even for providers (like DeepSeek) that
        # don't expose an embeddings endpoint at all.
        self.embeddings = get_embedder()

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            api_key=api_key,
            base_url=base_url,  # None => official OpenAI endpoint
        )
        # method="function_calling" is used instead of the default
        # strict-JSON-schema mode, since that stricter mode is an
        # OpenAI-only feature that non-OpenAI endpoints (DeepSeek, etc.)
        # don't support - function calling is broadly compatible.
        self.structured_llm = self.llm.with_structured_output(
            CandidateEvaluation, method="function_calling"
        )

        self.chunks: List[Document] = self._load_and_split(chunk_size, chunk_overlap)
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.retriever_k}
        )

    # ------------------------------------------------------------------ #
    # Loading + splitting
    # ------------------------------------------------------------------ #
    def _load_and_split(self, chunk_size: int, chunk_overlap: int) -> List[Document]:
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()

        for p in pages:
            p.metadata["source"] = self.source_name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(pages)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve_context(self, jd: str) -> str:
        """Multi-query retrieval: pull chunks relevant to the JD as a whole,
        plus chunks specifically about skills / experience / education, then
        dedupe. This gives the LLM a well-rounded, grounded view of the resume
        rather than only the single most JD-similar paragraph."""
        seen = set()
        collected: List[Document] = []

        for template in RETRIEVAL_QUERIES_TEMPLATE:
            query = template.format(jd=jd)
            for doc in self.retriever.invoke(query):
                key = doc.page_content[:120]
                if key not in seen:
                    seen.add(key)
                    collected.append(doc)

        if not collected:
            return "(No relevant content retrieved from resume.)"

        return "\n\n---\n\n".join(
            f"[chunk {i+1}]\n{doc.page_content}" for i, doc in enumerate(collected)
        )

    # ------------------------------------------------------------------ #
    # Evaluation (RAG generation step)
    # ------------------------------------------------------------------ #
    def evaluate(self, jd: str) -> CandidateEvaluation:
        context = self.retrieve_context(jd)
        chain = EVALUATION_PROMPT | self.structured_llm
        result: CandidateEvaluation = chain.invoke(
            {"jd": jd, "context": context, "source_name": self.source_name}
        )
        return result

    def answer_question(self, question: str) -> str:
        """Free-form RAG Q&A over this single resume (answers only from
        retrieved resume content)."""
        docs = self.retriever.invoke(question)
        context = "\n\n---\n\n".join(d.page_content for d in docs) or "(no content)"
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the question using ONLY the resume excerpts provided. "
                    "If the answer is not in the excerpts, say you don't have "
                    "enough information in the resume to answer.",
                ),
                ("human", "Resume excerpts:\n{context}\n\nQuestion: {question}"),
            ]
        )
        chain = qa_prompt | self.llm
        return chain.invoke({"context": context, "question": question}).content


# --------------------------------------------------------------------------- #
# 3. Convenience helpers used by the Streamlit app
# --------------------------------------------------------------------------- #
def save_uploaded_pdf(uploaded_file) -> str:
    """Persist a Streamlit UploadedFile to a temp path and return that path."""
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def build_engine(
    uploaded_file,
    api_key: str,
    llm_model: str = "deepseek-chat",
    base_url: Optional[str] = "https://api.deepseek.com",
) -> ResumeRAGEngine:
    path = save_uploaded_pdf(uploaded_file)
    name = os.path.splitext(uploaded_file.name)[0]
    return ResumeRAGEngine(
        pdf_path=path,
        source_name=name,
        api_key=api_key,
        llm_model=llm_model,
        base_url=base_url,
    )


def rank_candidates(
    evaluations: List[CandidateEvaluation],
) -> List[CandidateEvaluation]:
    """Sort candidates best-to-worst by match_score (ties broken by
    recommendation strength)."""
    rec_order = {"Strong Hire": 3, "Hire": 2, "Maybe": 1, "No Hire": 0}
    return sorted(
        evaluations,
        key=lambda e: (e.match_score, rec_order.get(e.recommendation, 0)),
        reverse=True,
    )
