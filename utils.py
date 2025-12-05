# utils.py

import os
import streamlit as st
from groq import Groq
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# SIDEBAR UI
# -----------------------------
def sidebar_setup():
    """
    Sidebar setup for API selection, API key input, and model selection.
    Returns: (api_choice, optional_api_key, engine)
    """

    # SAFELY load logo
    logo_path = os.path.join("static", "rc_logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=60)
    else:
        st.sidebar.warning("Logo missing: static/dj_logo.png")

    st.sidebar.title("API Key Selection")

    # Track API selection
    if "api_choice" not in st.session_state:
        st.session_state.api_choice = "None"

    api_choice = st.sidebar.selectbox(
        "Choose API:",
        ["None", "Groq", "OpenAI"],
        index=["None", "Groq", "OpenAI"].index(st.session_state.api_choice)
    )
    st.session_state.api_choice = api_choice

    # API key input
    optional_api_key = None
    if api_choice != "None":
        optional_api_key = st.sidebar.text_input(
            f"{api_choice} API Key",
            type="password",
            value=st.session_state.get("optional_api_key", "")
        )
        st.session_state.optional_api_key = optional_api_key

        if optional_api_key:
            st.sidebar.success(f"Using {api_choice} API.")
        else:
            st.sidebar.info(f"Please enter a valid {api_choice} API key.")
    else:
        st.session_state.optional_api_key = None

    # -----------------------------
    # Model selection
    # -----------------------------
    engine = None

    if api_choice == "Groq":
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.3-8b-instant",
            "mixtral-8x7b-32768",
            "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-qwen-32b",
        ]

        engine = st.sidebar.selectbox("Select Groq model:", groq_models)

    elif api_choice == "OpenAI":
        engine = st.sidebar.selectbox(
            "Select OpenAI model:",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
        )

    return api_choice, optional_api_key, engine


# -----------------------------
# EMBEDDINGS
# -----------------------------
def initialize_embeddings():
    """Initialize HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# -----------------------------
# LLM INITIALIZATION
# -----------------------------
def initialize_llm(api_choice, optional_api_key, engine, temperature=0.3, max_tokens=300):
    """
    Initialize the selected LLM.
    Groq → uses groq SDK (stable)
    OpenAI → uses langchain wrapper
    """

    # -----------------------------
    # GROQ LLM
    # -----------------------------
    if api_choice == "Groq" and optional_api_key:

        client = Groq(api_key=optional_api_key)

        def groq_chat(prompt):
            response = client.chat.completions.create(
                model=engine,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        return groq_chat

    # -----------------------------
    # OPENAI LLM
    # -----------------------------
    elif api_choice == "OpenAI" and optional_api_key:
        return ChatOpenAI(
            openai_api_key=optional_api_key,
            model=engine,
            temperature=temperature,
            max_tokens=max_tokens
        )

    return None
