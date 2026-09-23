import os
from pathlib import PurePath

import httpx
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


st.set_page_config(page_title="Student Study Assistant", page_icon="📚")
st.title("Student Study Assistant")
st.caption("Ask questions about the Class 12 Physics textbook.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict]) -> None:
    for source in sources:
        st.write(
            f"{source['source_file']}, page {source['page_number']} "
            f"({source['record_type']})"
        )
        if source.get("image_path"):
            image_name = PurePath(source["image_path"].replace("\\", "/")).name
            image_path = os.path.join("data", "pages", image_name)
            st.image(image_path, caption=f"Page {source['page_number']}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


if question := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[:-1]
    ]
    try:
        response = httpx.post(
            f"{API_URL}/query",
            json={"question": question, "chat_history": history},
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        st.error(f"API request failed: {exc}")
    else:
        assistant_message = {
            "role": "assistant",
            "content": payload["answer"],
            "sources": payload.get("sources", []),
        }
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.write(assistant_message["content"])
            render_sources(assistant_message["sources"])
