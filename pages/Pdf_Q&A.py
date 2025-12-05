import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from utils import sidebar_setup, initialize_llm


def main():
    load_dotenv()
    st.set_page_config(page_title="Conversational RAG With PDF Uploads", layout="wide")

    # TITLE
    st.title("📚 PDF Q&A Assistant with RAG")

    api_choice, optional_api_key, engine = sidebar_setup()
    llm = initialize_llm(api_choice, optional_api_key, engine)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    user_input = st.chat_input("Ask your question:")

    # Load Documents
    documents = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            docs = PyPDFLoader(temp_path).load()
            documents.extend(docs)

    # ----------------------------
    #       RAG PIPELINE
    # ----------------------------
    if documents and llm:

        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)

        # Vectorstore
        db = Chroma.from_documents(
            split_docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        retriever = db.as_retriever()

        # PROMPT
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
            You are a helpful assistant. Use ONLY the following context from uploaded PDFs.
            If answer not found in context, reply:
            “I couldn't find this answer in the uploaded PDFs.”
            \n\nContext:\n{context}
             """
             ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

        # LCEL CHAIN
        def get_context(query):
            docs = retriever.get_relevant_documents(query)
            return "\n\n".join([d.page_content for d in docs])

        rag_chain = (
            {
                "context": lambda x: get_context(x["question"]),
                "question": RunnablePassthrough(),
                "chat_history": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        # SESSION HISTORY
        if "history" not in st.session_state:
            st.session_state.history = []

        # SHOW HISTORY
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # PROCESS USER INPUT
        if user_input:
            st.session_state.history.append({"role": "user", "content": user_input})

            with st.spinner("Thinking..."):
                response = rag_chain.invoke({
                    "question": user_input,
                    "chat_history": st.session_state.history
                })

            answer = response.content
            st.session_state.history.append({"role": "assistant", "content": answer})

            # SHOW
            with st.chat_message("assistant"):
                st.write(answer)

    else:
        if not llm:
            st.write("⚠ Please select API & enter valid key in sidebar.")
        elif user_input:
            st.write("⚠ Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
