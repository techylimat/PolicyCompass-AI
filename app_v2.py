import streamlit as st
import os
import pysqlite3 as sqlite3
import sys
import traceback
import tempfile

# Set the pysqlite3 path for ChromaDB
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# --- UI elements ---
st.set_page_config(page_title="PolicyCompass AI 🤖", layout="wide")

# Custom CSS for a beautiful UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, .stApp {
        background-color: #1a1a2e;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    .st-emotion-cache-1c5v41f {
        background-color: #242442;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .st-emotion-cache-1629p8f {
        padding-top: 1rem;
    }

    /* Streamlit's chat message containers */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 0.5rem;
    }

    /* Specific user and assistant message styling */
    .user-message-bubble {
        background-color: #3b3b64;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        display: inline-block;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .assistant-message-bubble {
        background-color: #2a2a47;
        color: #e0e0e0;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        display: inline-block;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .st-emotion-cache-6q9r41 {
        background-color: #242442;
        color: #e0e0e0;
        border-radius: 25px;
        border: 1px solid #44446e;
        padding: 1rem 1.5rem;
    }
    .st-emotion-cache-13vn43k {
        background-color: #00796b;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 1rem;
    }
    .st-emotion-cache-1ghh0z9 {
        padding: 0.5rem;
        background-color: #2a2a47;
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .st-emotion-cache-h5g10 {
        padding: 0.5rem;
        background-color: #2a2a47;
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .st-emotion-cache-1ghh0z9 > div, .st-emotion-cache-h5g10 > div {
        color: white;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

# --- Main Logic ---
def create_rag_system(document_path):
    try:
        if not os.path.exists(document_path):
            st.error(f"Document not found at: {document_path}")
            return None

        # 1. Load Document
        st.write("Loading document...")
        loader = PyPDFLoader(document_path)
        docs = loader.load()

        # 2. Split Document into Chunks
        st.write("Splitting document...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # 3. Create Embeddings and Vector Store
        st.write("Creating vector store...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()
        st.write("Vector store created and persisted.")

        # 4. Create the Retrieval Chain
        st.write("Creating retrieval chain...")
        llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768",
            groq_api_key=os.environ.get("GROQ_API_KEY")
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert on data policy and regulations. Your role is to provide precise, factual answers to user questions based ONLY on the provided context. Do not make up any information. If you cannot find the answer, state that you cannot. The context is:\n\n{context}"),
            ("user", "{input}")
        ])

        # This chain combines the retrieved documents and the user query to get an initial answer
        rag_chain = (
            {"context": retriever, "input": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        st.write("Retrieval chain created.")
        return rag_chain

    except Exception as e:
        st.error(f"An error occurred during RAG system creation: {e}")
        st.error(traceback.format_exc())
        return None

def main():
    st.sidebar.title("PolicyCompass AI 🤖")
    st.sidebar.write("Get instant, verifiable guidance on data privacy policies.")

    uploaded_file = st.sidebar.file_uploader("Upload a PDF document to analyze", type="pdf")
    
    if uploaded_file is not None and st.session_state.rag_system is None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.rag_system = create_rag_system(tmp_file.name)
            os.remove(tmp_file.name)

    if st.session_state.rag_system is None and uploaded_file is not None:
        st.sidebar.error("Failed to initialize RAG system. Check console for details.")
    elif st.session_state.rag_system is not None:
        st.sidebar.success("✅ System is ready! Ask your question below.")

    st.markdown("<h1 style='text-align: center; color: #64ffda; padding: 1rem;'>💬 Chat with PolicyCompass AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a1a1c4;'>Upload a PDF on data policy or regulations to get started.</p>", unsafe_allow_html=True)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f"<div class='user-message-bubble'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message-bubble'>{message['content']}</div>", unsafe_allow_html=True)

    # Accept user input
    if prompt := st.chat_input("What is the GDPR's definition of 'personal data'?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(f"<div class='user-message-bubble'>{prompt}</div>", unsafe_allow_html=True)

        if st.session_state.rag_system is None:
            with st.chat_message("assistant"):
                st.markdown("<div class='assistant-message-bubble'>Please upload a document first to activate the system.</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": "Please upload a document first to activate the system."})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching and validating information..."):
                    try:
                        # 1. Get the initial response
                        initial_response = st.session_state.rag_system.invoke(prompt)

                        # --- This is the self-reflective and guardrail layer ---
                        # In a real Langraph implementation, this would be a node in the graph.
                        # We simulate it by creating a new prompt and LLM call.
                        
                        reflection_prompt = ChatPromptTemplate.from_messages([
                            ("system", "You are an expert auditor. Your task is to verify an AI's response against the original source context. Respond with 'PASS' if the answer is fully supported by the context. If it's not, explain why and provide a corrected answer based ONLY on the context. Do not make up new information."),
                            ("user", "Original Answer:\n{answer}\n\nContext:\n{context}")
                        ])

                        # This chain runs the reflection
                        reflection_chain = (
                            reflection_prompt
                            | st.session_state.rag_system.llm
                            | StrOutputParser()
                        )
                        
                        # Guardrail check 1: Prevent legal advice
                        if "advice" in prompt.lower() or "legal" in prompt.lower() or "consultant" in prompt.lower():
                            final_answer = "As a non-human entity, I cannot provide legal advice. Please consult with a qualified legal professional."
                        else:
                            # 2. Perform the self-reflection
                            reflection_result = reflection_chain.invoke({
                                "answer": initial_response,
                                "context": st.session_state.rag_system.retriever.invoke(prompt)
                            })

                            if "PASS" in reflection_result:
                                final_answer = initial_response
                            else:
                                final_answer = f"**Self-Correction Initiated:**\n\n{reflection_result}"
                        
                        # --- End of self-reflection and guardrail layer ---
                        
                        st.markdown(f"<div class='assistant-message-bubble'>{final_answer}</div>", unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})

                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        st.markdown(f"<div class='assistant-message-bubble'>{error_message}</div>", unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        print(traceback.format_exc())

if __name__ == "__main__":
    main()
