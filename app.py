import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar la API de Google
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("No se encontró la GOOGLE_API_KEY en las variables de entorno")

genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Configuración de la página y sidebar
st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

st.header("Chat with multiple PDFs :books:")

# Sidebar
with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)
            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            # create vector store
            vectorstore = get_vector_store(text_chunks)
            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)
        st.success("Processing complete!")

    # Agregar un mensaje de depuración
    if GOOGLE_API_KEY:
        st.write(f"Using API Key: {GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-5:]}")
    else:
        st.error("API Key not found!")

# Chat interface
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    handle_userinput(user_question)

# Plantillas HTML para los mensajes
user_template = '<div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;"><strong>Human:</strong> {{MSG}}</div>'
bot_template = '<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;"><strong>AI:</strong> {{MSG}}</div>'

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = None