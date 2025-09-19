import streamlit as st
from dotenv import load_dotenv
import tempfile
from datetime import datetime

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# -----------------------
# Function to load PDFs
# -----------------------
def load_uploaded_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents.extend(loader.load())  # loads all pages
    return documents

# -----------------------
# Main app
# -----------------------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    # Sidebar upload
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process PDFs'",
            type=['pdf'],
            accept_multiple_files=True
        )
        process_clicked = st.button("Process PDFs")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_history_for_chain" not in st.session_state:
        st.session_state.chat_history_for_chain = []

    # -----------------------
    # Determine theme for chat colors
    # -----------------------
    try:
        is_dark = st.get_option("theme.base") == "dark"
    except:
        is_dark = False  # fallback

    user_bg = "#0A2A6B" if is_dark else "#D3F2FF"
    user_color = "white" if is_dark else "black"

    bot_bg = "#333333" if is_dark else "#F0F0F0"
    bot_color = "white" if is_dark else "black"

    # -----------------------
    # Process PDFs
    # -----------------------
    if process_clicked and pdf_docs:
        with st.spinner("Processing PDFs..."):
            documents = load_uploaded_pdfs(pdf_docs)

            # Split text into chunks for retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            # Create embeddings + FAISS vectorstore
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # Strict PDF-only prompt
            PDF_QA_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a helpful assistant that answers questions using the provided PDF context. 

Context:
{context}

Question: {question}

Do not answer questions that are not related to the PDF content. If the question is unrelated, respond with "I can only answer questions related to the provided PDF content."
"""
            )

            # Create QA chain
            llm = ChatOpenAI(model="gpt-5-nano")
            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
                return_source_documents=False,
                combine_docs_chain_kwargs={"prompt": PDF_QA_PROMPT}
            )
        st.success("PDFs processed successfully!")

    # -----------------------
    # Chat box
    # -----------------------
    user_question = st.chat_input("Ask a question about your PDFs:")
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation.invoke({
            "question": user_question,
            "chat_history": st.session_state.chat_history_for_chain
        })

        # Update histories
        st.session_state.chat_history_for_chain.append(
            (user_question, response["answer"])
        )
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response["answer"],
        })

    # -----------------------
    # Display chat messages
    # -----------------------
    chat_container = st.container()
    for chat in st.session_state.chat_history:
        timestamp = datetime.now().strftime("%H:%M:%S")

        # User message
        with chat_container:
            st.markdown(
                f"<div style='background-color:{user_bg}; color:{user_color}; padding:10px; border-radius:8px; margin-bottom:4px;'>"
                f"<b>You:</b> {chat['question']} "
                f"<span style='float:right;font-size:10px;color:gray;'>{timestamp}</span>"
                f"</div>", unsafe_allow_html=True
            )

        # Assistant message
        with chat_container:
            st.markdown(
                f"<div style='background-color:{bot_bg}; color:{bot_color}; padding:10px; border-radius:8px; margin-bottom:12px;'>"
                f"<b>Bot:</b> {chat['answer']} "
                f"<span style='float:right;font-size:10px;color:gray;'>{timestamp}</span>"
                f"</div>", unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
