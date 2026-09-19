"""Streamlit front end.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polyglot.config import Config                      # noqa: E402
from polyglot.generation.prompts import LANGUAGE_NAMES   # noqa: E402
from polyglot.pipeline import ResearchAssistant          # noqa: E402

st.set_page_config(page_title="Polyglot Research Assistant", layout="wide")


@st.cache_resource(show_spinner="Loading index and models...")
def get_assistant(config_path: str):
    cfg = Config.load(config_path)
    return ResearchAssistant.load(cfg)


st.title("Polyglot")
st.caption("Ask in any language. Sources stay in their original language; the answer follows yours.")

with st.sidebar:
    st.header("Settings")
    config_path = st.text_input("Config file", value="config.yaml")
    top_k = st.slider("Passages in context", 3, 20, 8)
    languages = ["auto"] + sorted(LANGUAGE_NAMES)
    answer_language = st.selectbox(
        "Answer language", languages,
        format_func=lambda c: "Match the question" if c == "auto" else f"{LANGUAGE_NAMES[c]} ({c})",
    )
    show_retrieval = st.checkbox("Show retrieval diagnostics", value=False)

try:
    assistant = get_assistant(config_path)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Build an index first:  `polyglot ingest ./demo/corpus`")
    st.stop()

with st.sidebar:
    st.divider()
    st.metric("Chunks indexed", len(assistant.vector_store))
    st.caption(f"Embeddings: `{assistant.embedder.backend}`")
    if assistant.embedder.backend == "hashing":
        st.warning("Offline hashing encoder in use — retrieval quality is degraded.")

question = st.text_area("Question", height=90, placeholder="e.g. मूल्यांकन के लिए कौन से मापदंड प्रयोग किए गए?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and answering..."):
        answer = assistant.ask(
            question,
            k=top_k,
            answer_language=None if answer_language == "auto" else answer_language,
        )

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Answer")
        if answer.unsupported:
            st.warning("The indexed sources may not fully support this answer.")
        st.markdown(answer.text)
        st.caption(
            f"Question detected as `{answer.question_language}` · "
            f"answered in `{answer.answer_language}`"
        )

    with right:
        st.subheader("Sources")
        for citation in answer.citations:
            label = f"[{citation.marker}] {citation.title}"
            with st.expander(label):
                if citation.section:
                    st.caption(citation.section)
                st.caption(f"Language: {citation.language}")
                st.write(citation.snippet + "...")
                if citation.source_path:
                    st.code(citation.source_path, language=None)

    if show_retrieval:
        st.divider()
        st.subheader("Retrieval diagnostics")
        st.dataframe(
            [
                {
                    "rank": i,
                    "document": sc.chunk.title,
                    "section": sc.chunk.section,
                    "lang": sc.chunk.language,
                    "fused": round(sc.score, 4),
                    "dense_rank": sc.dense_rank,
                    "sparse_rank": sc.sparse_rank,
                    "rerank": None if sc.rerank_score is None else round(sc.rerank_score, 4),
                }
                for i, sc in enumerate(answer.contexts, start=1)
            ],
            use_container_width=True,
        )
