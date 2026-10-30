import os
import hashlib
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from openai import OpenAI
from streamlit.errors import StreamlitSecretNotFoundError

CHUNK, OVERLAP, TOP_K = 800, 150, 5
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

SYSTEM = (
    "You answer questions using ONLY the numbered context snippets provided. "
    "Cite snippets inline like [1], [2]. If the answer is not in the context, "
    "say \"I couldn't find that in the documents.\" Do not use outside knowledge."
)


@st.cache_resource
def get_collection():
    emb = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient("chroma_db")
    return client.get_or_create_collection("docs", embedding_function=emb)


@st.cache_resource
def get_llm():
    try:
        api_key = os.getenv("LLM_API_KEY") or st.secrets["LLM_API_KEY"]
    except (KeyError, StreamlitSecretNotFoundError) as error:
        raise RuntimeError("Set LLM_API_KEY in the environment or Streamlit secrets.") from error

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    )


def chunk_text(text):
    step = CHUNK - OVERLAP
    return [text[i:i + CHUNK] for i in range(0, len(text), step) if text[i:i + CHUNK].strip()]


def ingest(file, col):
    reader = PdfReader(file)
    ids, docs, metas = [], [], []
    for page_no, page in enumerate(reader.pages, start=1):
        for i, c in enumerate(chunk_text(page.extract_text() or "")):
            ids.append(hashlib.md5(f"{file.name}-{page_no}-{i}".encode()).hexdigest())
            docs.append(c)
            metas.append({"source": file.name, "page": page_no})
    if docs:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def answer(question, col):
    res = col.query(query_texts=[question], n_results=TOP_K)
    docs, metas = res["documents"][0], res["metadatas"][0]
    context = "\n\n".join(
        f"[{i}] ({m['source']}, p.{m['page']})\n{d}" for i, (d, m) in enumerate(zip(docs, metas), 1)
    )
    resp = get_llm().chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content, list(zip(docs, metas))


st.set_page_config(page_title="Doc Assistant", page_icon="📄")
st.title("📄 RAG Document Assistant")
col = get_collection()

with st.sidebar:
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Index documents"):
        with st.spinner("Indexing..."):
            n = sum(ingest(f, col) for f in files)
        st.success(f"Indexed {n} chunks")
    st.caption(f"Chunks in store: {col.count()}")

if "history" not in st.session_state:
    st.session_state.history = []

for role, text, sources in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            with st.expander("Sources"):
                for i, (d, m) in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {m['source']} — p.{m['page']}**\n\n{d}")

if q := st.chat_input("Ask about your documents"):
    st.session_state.history.append(("user", q, None))
    with st.chat_message("user"):
        st.markdown(q)
    if col.count() == 0:
        reply, src = "Upload and index a PDF first.", None
    else:
        with st.spinner("Thinking..."):
            reply, src = answer(q, col)
    st.session_state.history.append(("assistant", reply, src))
    st.rerun()
