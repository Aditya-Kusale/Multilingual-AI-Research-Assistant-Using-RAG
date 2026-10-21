"""Streamlit front end for the hosted multilingual research demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polyglot.config import Config  # noqa: E402
from polyglot.generation.prompts import LANGUAGE_NAMES  # noqa: E402
from polyglot.pipeline import ResearchAssistant  # noqa: E402

st.set_page_config(
    page_title="Polyglot · Multilingual Research Assistant",
    page_icon="🌐",
    layout="wide",
)

# Secrets are configured in Streamlit Cloud's app settings. Only the keys the
# LLM client actually needs are copied into the process environment.
for secret_name in (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "POLYGLOT__GENERATION__PROVIDER",
    "POLYGLOT__GENERATION__MODEL",
):
    try:
        if secret_name in st.secrets:
            os.environ[secret_name] = str(st.secrets[secret_name])
    except Exception:
        # Local runs without a secrets.toml are supported.
        pass

if "POLYGLOT__GENERATION__PROVIDER" not in os.environ:
    if os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["POLYGLOT__GENERATION__PROVIDER"] = "anthropic"
    elif os.environ.get("OPENAI_API_KEY"):
        os.environ["POLYGLOT__GENERATION__PROVIDER"] = "openai"


@st.cache_resource(show_spinner=False)
def get_assistant(config_path: str) -> ResearchAssistant:
    cfg = Config.load(config_path)
    if not cfg.storage.index_dir.is_absolute():
        cfg.storage.index_dir = (REPOSITORY_ROOT / cfg.storage.index_dir).resolve()

    index_file = cfg.storage.index_dir / "chunks.jsonl"
    if index_file.exists():
        return ResearchAssistant.load(cfg)

    # Build a small, multilingual sample index on first launch. This makes a
    # fresh Streamlit deployment usable without a separate CLI/bootstrap step.
    assistant = ResearchAssistant(cfg)
    assistant.ingest([PROJECT_ROOT / "demo" / "corpus"])
    return assistant


st.title("🌐 Polyglot")
st.caption(
    "Ask research questions across English, Hindi, French, and Chinese. "
    "Retrieve original-language evidence and inspect every citation."
)

with st.sidebar:
    st.header("Research settings")
    top_k = st.slider("Sources in context", 3, 12, 6)
    language_codes = ["auto"] + sorted(LANGUAGE_NAMES)
    answer_language = st.selectbox(
        "Answer language",
        language_codes,
        format_func=lambda code: (
            "Match the question" if code == "auto"
            else f"{LANGUAGE_NAMES[code]} ({code})"
        ),
    )
    show_retrieval = st.checkbox("Show retrieval diagnostics", value=False)
    st.divider()
    st.caption("Sample corpus · English · Hindi · French · Chinese")

config_path = str(PROJECT_ROOT / "streamlit_config.yaml")
try:
    with st.spinner("Preparing the multilingual sample index on first launch…"):
        assistant = get_assistant(config_path)
except Exception as exc:
    st.error("The research workspace could not start.")
    st.exception(exc)
    st.stop()

st.caption(
    f"{len(assistant.vector_store)} indexed passages · "
    f"retrieval: {assistant.embedder.backend}"
)
if assistant.embedder.backend == "hashing":
    st.warning(
        "Semantic embeddings are unavailable; retrieval is running in fallback "
        "mode and may be less accurate."
    )

if assistant.cfg.generation.provider == "echo":
    st.info(
        "Demo mode uses extractive, citation-linked answers. To enable generated "
        "answers, add ANTHROPIC_API_KEY or OPENAI_API_KEY in Streamlit app secrets."
    )

with st.form("research-question", clear_on_submit=False):
    question = st.text_area(
        "Research question",
        height=100,
        placeholder="For example: मूल्यांकन के लिए कौन से मापदंड प्रयोग किए गए?",
    )
    submitted = st.form_submit_button("Search the research corpus", type="primary")

if submitted:
    if not question.strip():
        st.info("Enter a question to search the sample corpus.")
    else:
        try:
            with st.spinner("Searching across languages and assembling citations…"):
                answer = assistant.ask(
                    question.strip(),
                    k=top_k,
                    answer_language=(
                        None if answer_language == "auto" else answer_language
                    ),
                )
            if answer.unsupported:
                st.warning("The indexed sources may not fully support this answer.")

            answer_column, sources_column = st.columns([3, 2], gap="large")
            with answer_column:
                st.subheader("Answer")
                st.markdown(answer.text)
                st.caption(
                    f"Question detected as {answer.question_language}; "
                    f"answer language: {answer.answer_language}."
                )
            with sources_column:
                st.subheader("Sources")
                if not answer.citations:
                    st.info("No citations were returned for this question.")
                for citation in answer.citations:
                    with st.expander(f"[{citation.marker}] {citation.title}"):
                        if citation.section:
                            st.caption(citation.section)
                        st.caption(f"Source language: {citation.language}")
                        st.write(citation.snippet + "…")
                        if citation.source_path:
                            st.code(citation.source_path, language=None)

            if show_retrieval:
                st.divider()
                st.subheader("Retrieval diagnostics")
                st.dataframe(
                    [
                        {
                            "rank": rank,
                            "document": item.chunk.title,
                            "section": item.chunk.section,
                            "language": item.chunk.language,
                            "fused score": round(item.score, 4),
                            "dense rank": item.dense_rank,
                            "sparse rank": item.sparse_rank,
                            "rerank score": (
                                None if item.rerank_score is None
                                else round(item.rerank_score, 4)
                            ),
                        }
                        for rank, item in enumerate(answer.contexts, start=1)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as exc:
            st.error("The search request failed. Check the app logs for details.")
            st.exception(exc)
