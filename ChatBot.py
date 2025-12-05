# ChatBot.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils import sidebar_setup, initialize_llm

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a cheerful and friendly chatbot. Please respond to the user's queries."),
        ("user", "Question:{question}")
    ]
)

# Generate response based on the selected API
def generate_response(question, api_key, engine, api_choice, temperature, max_tokens):
    if api_choice == "Groq":
        llm = ChatGroq(groq_api_key=api_key, model_name=engine, streaming=True)
    elif api_choice == "OpenAI":
        import openai
        openai.api_key = api_key
        llm = ChatOpenAI(model=engine, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError("Invalid API choice.")
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({'question': question})

# Main Streamlit app
def main():
    st.set_page_config(page_title="🤖 Friendly AI Chatbot", layout="wide")

    # Load CSS
    with open("static/styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    # Display chatbot title
    st.markdown(
        """
        <div class="title-box">
            <h1>🤖 Friendly AI Chatbot 🐶</h1>
            <h3>Bringing you answers with speed, smarts, and a smile!</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Sidebar setup for API and model selection
    api_choice, optional_api_key, engine = sidebar_setup()

    # Sliders for Temperature and Max Tokens
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input
    if user_input := st.chat_input("Your Question:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        if api_choice != "None" and optional_api_key:
            try:
                if not engine:
                    st.error("Please select a valid engine.")
                    return

                # Spinner while generating the response
                with st.spinner("Thinking... 🤔"):
                    response = generate_response(user_input, optional_api_key, engine, api_choice, temperature, max_tokens)

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Select an API and provide a valid key.")


if __name__ == "__main__":
    main()