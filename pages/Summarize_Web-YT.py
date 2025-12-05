import streamlit as st
import validators
from dotenv import load_dotenv

from yt_dlp import YoutubeDL
from utils import sidebar_setup, initialize_llm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import UnstructuredURLLoader


def main():
    load_dotenv()
    st.set_page_config(page_title="Summarize Text From YT or Website", page_icon="📝", layout="wide")

    # Load CSS
    with open("static/styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    api_choice, optional_api_key, engine = sidebar_setup()

    st.markdown(
        """
        <div class="title-box">
            <h1>📋 Simplify Your Learning ▶️</h1>
            <h3>Paste a YouTube or Website URL to get a 300-word summary.</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    url = st.text_input("", placeholder="Paste a YouTube or Website URL here...")

    llm = initialize_llm(api_choice, optional_api_key, engine)

    # NEW LANGCHAIN 1.1 SUMMARY PROMPT
    prompt = ChatPromptTemplate.from_template("""
        Summarize the following content in about 300 words:

        {text}
    """)

    # NEW LCEL SUMMARY CHAIN
    summarize_chain = (
        {"text": RunnablePassthrough()}
        | prompt
        | llm
    )

    def fetch_youtube_transcript(url):
        try:
            ydl_opts = {"quiet": True, "noplaylist": True}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get("description", "")
        except Exception as e:
            st.error(f"Error fetching YouTube transcript: {e}")
            return None

    if st.button("Summarize Content"):

        if not optional_api_key or not url.strip():
            st.error("Please enter API key + valid URL.")
            return

        if not validators.url(url):
            st.error("Invalid URL. Please enter a correct link.")
            return

        if not llm:
            st.error("Please choose API and enter key in sidebar.")
            return

        try:
            with st.spinner("Fetching content..."):

                if "youtube.com" in url or "youtu.be" in url:
                    content = fetch_youtube_transcript(url)
                    if not content:
                        st.error("No transcript found.")
                        return
                    text_data = content

                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False)
                    docs = loader.load()
                    text_data = "\n\n".join([d.page_content for d in docs])

            with st.spinner("Generating summary..."):
                result = summarize_chain.invoke(text_data)

            st.success(result.content)

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
