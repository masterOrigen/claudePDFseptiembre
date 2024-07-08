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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                 temperature=0.3,
                                 max_output_tokens=4096)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola, he procesado tus documentos PDF. ¿Qué te gustaría saber sobre ellos?"}]
    st.session_state.conversation = None

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"Human: {message.content}")
        else:
            st.write(f"AI: {message.content}")

def main():
    st.set_page_config(page_title="Chatbot PDF con Gemini", page_icon="📚")

    with st.sidebar:
        st.title("Menú:")
        pdf_docs = st.file_uploader(
            "Sube tus archivos PDF y haz clic en 'Procesar'", accept_multiple_files=True)
        if st.button("Procesar"):
            with st.spinner("Procesando documentos..."):
                raw_text = get_pdf_text(pdf_docs)
                st.sidebar.write(f"Texto extraído (primeros 500 caracteres): {raw_text[:500]}...")
                text_chunks = get_text_chunks(raw_text)
                st.sidebar.write(f"Número de chunks creados: {len(text_chunks)}")
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success(f"¡Documentos procesados con éxito! Se crearon {len(text_chunks)} chunks.")
                clear_chat_history()

    st.title("Chatbot PDF con Gemini 📚")
    st.write("Bienvenido al chatbot. ¿Qué te gustaría saber sobre los documentos?")
    st.sidebar.button('Limpiar historial de chat', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hola, sube algunos PDFs y hazme preguntas sobre ellos."}]

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if st.session_state.conversation is not None:
        user_question = st.text_input("Haz una pregunta sobre tus documentos:")
        if user_question:
            handle_user_input(user_question)
    else:
        st.write("Por favor, sube y procesa algunos documentos PDF primero.")

if __name__ == "__main__":
    main()