import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

def sidebar_setup():
    """
    Sidebar setup for API selection, API key input, and model selection.
    """
    
    # Display logo and sidebar title
    st.sidebar.image("static/rc_logo.png", width=60)   
    st.sidebar.title("API Key Selection")           
        
    # API choice selection
    if "api_choice" not in st.session_state:
        st.session_state.api_choice = "None"  # Default value

    api_choice = st.sidebar.selectbox(
        "Choose API:", ["None", "Groq", "OpenAI"],
        index=["None", "Groq", "OpenAI"].index(st.session_state.api_choice)
    )
    st.session_state.api_choice = api_choice

    # API Key input with session_state persistence
    if api_choice != "None":
        if "optional_api_key" not in st.session_state:
            st.session_state.optional_api_key = ""  

        # Store API Key in session_state and use `key` parameter
        optional_api_key = st.sidebar.text_input(
            f"{api_choice} API Key", 
            type="password", 
            value=st.session_state.optional_api_key, 
            key="api_key_input"
        )

        # Update session_state with the entered API Key
        st.session_state.optional_api_key = optional_api_key

        # Display success or warning messages
        if optional_api_key:
            st.sidebar.success(f"Using {api_choice} API.")
        else:
            st.sidebar.info(f"Please enter a valid {api_choice} API key.")
    else:
        st.session_state.optional_api_key = None
        st.sidebar.info("No API selected.")

    # Model selection
    engine = None
    if api_choice == "Groq":
        engine = st.sidebar.selectbox(
            "Select Groq model:", ["llama-3.3-70b-versatile", "llama-3.3-8b-instant"]
        )
    elif api_choice == "OpenAI":
        engine = st.sidebar.selectbox(
            "Select OpenAI model:", ["gpt-3.5-turbo", "gpt-4"]
        )

    return api_choice, st.session_state.optional_api_key, engine   

    


def initialize_embeddings():
    """Initialize HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def initialize_llm(api_choice, optional_api_key, engine):
    """Initialize the selected LLM based on API choice and API key."""
    if api_choice == "Groq" and optional_api_key:
        return ChatGroq(groq_api_key=optional_api_key, model_name=engine)
    elif api_choice == "OpenAI" and optional_api_key:
        return ChatOpenAI(openai_api_key=optional_api_key, model=engine)
    return None