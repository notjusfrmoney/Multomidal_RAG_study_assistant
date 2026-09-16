import os
from pathlib import PurePath

import httpx
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


st.set_page_config(page_title="Student Study Assistant", page_icon="📚")
st.title("Student Study Assistant")
st.caption("Ask questions about the Class 12 Physics textbook.")
question = st.text_input("Ask a question")

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            response = httpx.post(
                f"{API_URL}/query",
                json={"question": question},
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            st.error(f"API request failed: {exc}")
        else:
            st.subheader("Answer")
            st.write(payload["answer"])
            st.subheader("Sources")
            for source in payload["sources"]:
                st.write(
                    f"- {source['source_file']}, page {source['page_number']} "
                    f"({source['record_type']})"
                )
                if source["image_path"]:
                    image_name = PurePath(source["image_path"].replace("\\", "/")).name
                    image_path = os.path.join("data", "pages", image_name)
                    st.image(image_path, caption=f"Page {source['page_number']}")
