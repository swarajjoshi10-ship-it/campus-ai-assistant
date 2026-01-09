import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from htmlTemplates import bot_template, user_template, css

load_dotenv(override=True)

# -----------------------
# Level 1: Basic LLM
# -----------------------
class FreeLLM:
    def __init__(self):
        self.generator = pipeline("text2text-generation", model="google/flan-t5-small")

    def __call__(self, prompt):
        return self.generator(prompt, max_length=200)[0]['generated_text']

# -----------------------
# PDF Processing & RAG
# -----------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# -----------------------
# Conversation & Retrieval
# -----------------------
chat_history = []

def get_conversation_chain(vector_store):
    llm = FreeLLM()

    def conversation(user_question):
        # Retrieve relevant chunks
        docs = vector_store.similarity_search(user_question, k=4)
        context = "\n".join([doc.page_content for doc in docs])

        # Generate answer
        prompt = f"Answer the question based on the context:\n{context}\nQuestion: {user_question}"
        answer = llm(prompt)

        # Save to session memory
        chat_history.append({"role": "user", "content": user_question})
        chat_history.append({"role": "bot", "content": answer})
        return {"answer": answer, "chat_history": chat_history}

    return conversation

# -----------------------
# Streamlit Interface
# -----------------------
def handle_userinput(user_question):
    response = st.session_state.conversation(user_question)
    for i, message in enumerate(response["chat_history"]):
        template = user_template if message["role"] == "user" else bot_template
        st.write(template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Campus Life Assistant", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("AI Campus Assistant")
    user_question = st.text_input("Ask a question about campus life:")

    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload Campus Documents (FAQs, Rules, Schedules)")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("Documents processed! You can now ask questions.")

if __name__ == "__main__":
    main()
