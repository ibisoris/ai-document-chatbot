import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
MAX_FILENAME_DISPLAY_LENGTH = 34


def configure_page() -> None:
    """Set Streamlit page metadata."""
    st.set_page_config(
        page_title="AI Document Chatbot",
        layout="wide",
    )


def initialize_session_state() -> None:
    """Create default state values used by the Streamlit interface."""
    defaults = {
        "active_document": None,
        "documents": [],
        "retrieved_sections": [],
        "answer": "",
        "last_question": "",
        "uploaded_file_id": None,
        "uploader_key": 0,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def api_url(path: str) -> str:
    """Build a full backend URL from an API path."""
    return f"{API_BASE_URL.rstrip('/')}{path}"


def format_api_error(error: requests.RequestException) -> str:
    """Convert backend/network failures into helpful UI messages."""
    if isinstance(error, requests.ConnectionError):
        return (
            "The FastAPI backend is not running. Start it with "
            "`uvicorn backend.main:app --reload`, then try again."
        )

    response = getattr(error, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail")
            if detail:
                return str(detail)
        except ValueError:
            pass

    return "Something went wrong while contacting the backend."


def format_filename(filename: str, max_length: int = MAX_FILENAME_DISPLAY_LENGTH) -> str:
    """Shorten long filenames for compact sidebar display."""
    if len(filename) <= max_length:
        return filename

    suffix_length = 10
    prefix_length = max_length - suffix_length - 3
    return f"{filename[:prefix_length]}...{filename[-suffix_length:]}"


def load_documents() -> None:
    """Fetch indexed document metadata from the backend."""
    try:
        response = requests.get(api_url("/documents"), timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        st.sidebar.warning(format_api_error(error))
        return

    st.session_state["documents"] = response.json()

    if st.session_state["documents"] and not st.session_state["active_document"]:
        st.session_state["active_document"] = st.session_state["documents"][-1]


def upload_document(uploaded_file) -> None:
    """Send a document upload to the FastAPI backend for indexing."""
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    try:
        response = requests.post(
            api_url("/upload-document"),
            files=files,
            timeout=180,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        st.sidebar.error(format_api_error(error))
        return

    document = response.json()
    st.session_state["active_document"] = document
    st.session_state["answer"] = ""
    st.session_state["last_question"] = ""
    st.session_state["retrieved_sections"] = []
    load_documents()
    st.sidebar.success("Upload complete. The document is ready to chat with.")


def clear_document() -> None:
    """Delete the active document from the backend and reset frontend state."""
    active_document = st.session_state.get("active_document")

    if active_document:
        document_id = active_document["document_id"]
        try:
            response = requests.delete(
                api_url(f"/documents/{document_id}"),
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            st.sidebar.error(format_api_error(error))
            return

    st.session_state["active_document"] = None
    st.session_state["retrieved_sections"] = []
    st.session_state["answer"] = ""
    st.session_state["last_question"] = ""
    st.session_state["uploaded_file_id"] = None
    st.session_state["uploader_key"] += 1
    load_documents()


def ask_question(question: str) -> None:
    """Send the user question to the backend RAG endpoint."""
    active_document = st.session_state.get("active_document")

    if not active_document:
        st.warning("Upload a document before asking a question.")
        return

    payload = {
        "question": question,
        "document_id": active_document["document_id"],
        "top_k": 3,
    }

    try:
        with st.spinner("Thinking with the uploaded document..."):
            response = requests.post(
                api_url("/ask-question"),
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
    except requests.RequestException as error:
        st.error(format_api_error(error))
        return

    result = response.json()
    st.session_state["last_question"] = question
    st.session_state["answer"] = result["answer"]
    st.session_state["retrieved_sections"] = result["retrieved_sections"]


def render_sidebar() -> None:
    """Render upload, document metadata, and reset controls."""
    with st.sidebar:
        st.title("Document")
        st.caption(f"Backend: `{API_BASE_URL}`")

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "txt", "csv", "json", "xlsx", "xls"],
            accept_multiple_files=False,
            key=f"pdf_uploader_{st.session_state['uploader_key']}",
        )

        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            if file_id != st.session_state["uploaded_file_id"]:
                st.session_state["uploaded_file_id"] = file_id
                upload_document(uploaded_file)

        if st.button("Clear document", use_container_width=True):
            clear_document()
            st.rerun()

        st.divider()
        st.subheader("Status")

        active_document = st.session_state.get("active_document")
        if active_document:
            st.success("Ready")
            st.write("**Filename:**")
            st.caption(format_filename(active_document["filename"]))
            st.write(f"**Type:** {active_document.get('document_kind', 'document')}")
            if active_document.get("page_count") is not None:
                st.write(f"**Pages:** {active_document['page_count']}")
            if active_document.get("record_count") is not None:
                st.write(f"**Records:** {active_document['record_count']}")
            st.write(f"**Chunks:** {active_document['chunk_count']}")
            st.write(f"**Upload status:** {active_document['status']}")
        else:
            st.info("No active document.")

        if st.button("Refresh documents", use_container_width=True):
            load_documents()
            st.rerun()

        if st.session_state["documents"]:
            st.divider()
            st.subheader("Indexed Documents")
            for document in st.session_state["documents"]:
                label = format_filename(document["filename"])
                if st.button(label, key=document["document_id"]):
                    st.session_state["active_document"] = document
                    st.session_state["answer"] = ""
                    st.session_state["last_question"] = ""
                    st.session_state["retrieved_sections"] = []
                    st.rerun()


def render_chat_area() -> None:
    """Render the main chatbot interface."""
    st.title("AI Document Chatbot")
    st.caption("Streamlit frontend -> FastAPI backend -> RAG services -> ChromaDB/OpenAI")

    active_document = st.session_state.get("active_document")
    if active_document:
        st.info(
            f"Chatting with **{active_document['title']}** "
            f"({active_document.get('document_kind', 'document')}, "
            f"{active_document['chunk_count']} chunks)."
        )
    else:
        st.info("Upload a PDF, TXT, CSV, JSON, or Excel file in the sidebar.")

    if st.session_state["last_question"]:
        with st.chat_message("user"):
            st.write(st.session_state["last_question"])

    with st.chat_message("assistant"):
        if st.session_state["answer"]:
            st.write(st.session_state["answer"])
        else:
            st.write("Upload a document or dataset, then ask a question about it.")

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question",
            placeholder="What skills are mentioned in this document?",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        ask_question(question.strip())
        st.rerun()


def render_retrieval_section() -> None:
    """Show retrieved context chunks inside an optional expander."""
    retrieved_sections = st.session_state["retrieved_sections"]

    if not retrieved_sections:
        return

    with st.expander("Show relevant document sections"):
        for index, result in enumerate(retrieved_sections, start=1):
            metadata = result.get("metadata") or {}
            chunk_number = metadata.get("chunk_index", index)
            title = metadata.get("document_title", "Uploaded Document")

            with st.container(border=True):
                st.subheader(f"Relevant Section {index}")
                st.markdown(f"**Document:** {title}")
                st.caption(
                    f"Chunk {chunk_number} from "
                    f"{metadata.get('source', 'document')}"
                )
                st.write(result["text"])
                st.caption(
                    f"Similarity distance: {result['distance']:.4f} "
                    "lower is closer"
                )


def main() -> None:
    """Run the Streamlit frontend."""
    configure_page()
    initialize_session_state()

    if not st.session_state["documents"]:
        load_documents()

    render_sidebar()
    render_chat_area()
    render_retrieval_section()


if __name__ == "__main__":
    main()
