import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
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

def get_conversational_chain():
    prompt_template = """
    Basándote en la siguiente información, proporciona una respuesta detallada y extensa a la pregunta. 
    Utiliza todos los detalles relevantes del contexto para elaborar una respuesta completa y exhaustiva.
    Si la información no está directamente en el contexto, intenta inferir o extrapolar basándote en lo que sabes.
    Si realmente no tienes suficiente información para responder, indica que necesitas más detalles para proporcionar una respuesta precisa.

    Contexto: {context}

    Pregunta: {question}

    Respuesta detallada:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   max_output_tokens=4096,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola, he procesado tus documentos PDF. ¿Qué te gustaría saber sobre ellos?"}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question, k=5)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response

def main():
    st.set_page_config(page_title="Chatbot PDF con Gemini", page_icon="📚")

    with st.sidebar:
        st.title("Menú:")
        pdf_docs = st.file_uploader(
            "Sube tus archivos PDF y haz clic en 'Procesar'", accept_multiple_files=True)
        if st.button("Procesar"):
            with st.spinner("Procesando documentos..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("¡Documentos procesados con éxito!")
                clear_chat_history()

    st.title("Chatbot PDF con Gemini 📚")
    st.write("Bienvenido al chatbot. ¿Qué te gustaría saber sobre los documentos?")
    st.sidebar.button('Limpiar historial de chat', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hola, sube algunos PDFs y hazme preguntas sobre ellos."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = user_input(prompt)
                full_response = response['output_text']
                st.write(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()