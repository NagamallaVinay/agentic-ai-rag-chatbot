import os
import sys

# Add the project root to Python's import path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from src.graph import run_query


st.set_page_config(
    page_title="Agentic AI RAG",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI Knowledge Assistant")
st.caption("LangGraph + Pinecone + Gemini RAG")

st.write(
    "Ask questions about Agentic AI. "
    "The assistant retrieves relevant knowledge-base content, "
    "generates a grounded answer, and provides a confidence score."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "confidence" in message:
            st.caption(
                f"Confidence score: {message['confidence']:.2f}"
            )

            with st.expander("Retrieved context"):
                for i, chunk in enumerate(
                    message["context"], start=1
                ):
                    st.markdown(f"**Chunk {i}**")
                    st.write(chunk)


query = st.chat_input("Ask a question about Agentic AI...")

if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            result = run_query(query)

        answer = result["final_answer"]
        confidence = result["confidence_score"]
        context = result["retrieved_context_chunks"]

        st.markdown(answer)
        st.caption(f"Confidence score: {confidence:.2f}")

        with st.expander("Retrieved context"):
            for i, chunk in enumerate(context, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(chunk)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "context": context,
        }
    )