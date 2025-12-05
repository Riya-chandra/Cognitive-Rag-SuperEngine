# ChatBot.py
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils import sidebar_setup, initialize_llm

load_dotenv()

# Chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a cheerful and friendly chatbot. Please respond to the user's queries."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, engine, api_choice, temperature=0.7, max_tokens=150):
    """
    Create llm based on api_choice and invoke the chain.
    Returns a plain string.
    """
    if api_choice == "Groq":
        llm = initialize_llm("Groq", api_key, engine, streaming=True, temperature=temperature, max_tokens=max_tokens)
    elif api_choice == "OpenAI":
        llm = initialize_llm("OpenAI", api_key, engine, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError("Invalid API choice.")

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Some LangChain versions expect dict input for the prompt variables
    result = chain.invoke({"question": question})
    # result might be an object; convert to string safely
    if hasattr(result, "content"):
        return result.content
    return str(result)

def main():
    st.set_page_config(page_title="🤖 Friendly AI Chatbot", layout="wide")

    # Load CSS (optional)
    css_path = "static/styles.css"
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="title-box">
            <h1>🤖 Friendly AI Chatbot 🐶</h1>
            <h3>Bringing you answers with speed, smarts, and a smile!</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    api_choice, optional_api_key, engine = sidebar_setup()

    # Temperature and max tokens
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input("Your Question:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        if api_choice != "None" and optional_api_key:
            if not engine:
                st.error("Please select an engine in the sidebar.")
            else:
                try:
                    with st.spinner("Thinking... 🤔"):
                        response = generate_response(user_input, optional_api_key, engine, api_choice, temperature, max_tokens)

                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.chat_message("assistant").write(response)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Select an API and provide a valid key in the sidebar.")

if __name__ == "__main__":
    main()
