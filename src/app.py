import json
import streamlit as st

from src.graph import run_query


st.set_page_config(
    page_title="Agentic AI RAG",
    page_icon="🤖",
    layout="wide",
)


st.title("🤖 Agentic AI Knowledge Assistant")
st.caption("LangGraph + Pinecone + Gemini RAG")

st.markdown(
    """
    Ask questions about the **Agentic AI eBook**.
    Answers are generated only from the retrieved knowledge-base context.
    """
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "result" in message:
            result = message["result"]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Confidence",
                    f"{result['confidence_score']:.0%}",
                )

            with col2:
                st.metric(
                    "Retrieved Chunks",
                    len(result["retrieved_context_chunks"]),
                )

            with st.expander("Retrieved Context"):
                for i, chunk in enumerate(
                    result["retrieved_context_chunks"],
                    start=1,
                ):
                    st.markdown(f"**Chunk {i}**")
                    st.write(chunk)


query = st.chat_input("Ask about Agentic AI...")


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
        with st.spinner("Searching the Agentic AI knowledge base..."):
            result = run_query(query)

        st.markdown(result["final_answer"])

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Confidence",
                f"{result['confidence_score']:.0%}",
            )

        with col2:
            st.metric(
                "Retrieved Chunks",
                len(result["retrieved_context_chunks"]),
            )

        with st.expander("Retrieved Context"):
            for i, chunk in enumerate(
                result["retrieved_context_chunks"],
                start=1,
            ):
                st.markdown(f"**Chunk {i}**")
                st.write(chunk)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["final_answer"],
                "result": result,
            }
        )