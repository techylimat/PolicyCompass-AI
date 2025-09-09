# PolicyCompass-AI

This project is a sophisticated Retrieval-Augmented Generation (RAG) system built to provide accurate and auditable guidance on complex data privacy policies and legal documents. It goes beyond a standard RAG implementation by integrating a self-reflective loop and robust guardrails to ensure accuracy and prevent the generation of misleading information.

The final application is a user-friendly chat interface built with Streamlit, allowing users to upload a document and ask complex policy questions.

## Core Innovation
The key innovation of this solution is its ability to self-correct. After an initial answer is generated, the system performs a self-audit against the original source context. This process ensures that the final response is fully supported by the document, thereby drastically reducing the risk of hallucinations and making the system a trustworthy tool for compliance professionals.

## Tech Stack
Framework: LangChain

User Interface: Streamlit

Document Loader: PyPDFLoader

Text Splitter: RecursiveCharacterTextSplitter

Embedding Model: HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)

Vector Store: ChromaDB (using pysqlite3)

LLM: mixtral-8x7b-32768 from Groq

## Project Files
app_v2.py: The main Streamlit application, containing all the core logic, including document processing, RAG chain creation, and the self-reflective loop.

requirements.txt: A list of all necessary Python libraries for deployment.

## Setup Instructions
Prepare Your Project: Ensure you have the following files in your project directory: app_v2.py and requirements.txt.

Set Up Groq API Key:

Sign up for a free account on Groq.

Go to "API Keys" and create a new key.

Set the API key as an environment variable named GROQ_API_KEY.

Run the Streamlit Application:

streamlit run app_v2.py


Your web browser should automatically open the application, ready for you to upload a PDF.

## Example Queries
"What is the GDPR's definition of 'personal data'?"

"What are the specific consent requirements for processing special categories of data?"

"Can a company transfer personal data to a country outside the EU under GDPR?"

## File for Dependencies
   
streamlit
langchain
langchain-community
langchain-groq
pysqlite3
pdfplumber
