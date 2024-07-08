import os
from PyPDF2 import PdfReader
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_doc):
    pdf_reader = PdfReader(pdf_doc)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_gemini_response(question, pdf_content):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = f"""
    Eres un asistente experto en análisis de documentos PDF. Se te ha proporcionado el contenido completo de un documento PDF.
    Utiliza toda la información disponible en el documento para responder la siguiente pregunta de manera detallada y precisa.
    Si la información no está en el documento, indica que no puedes encontrar esa información específica en el PDF proporcionado.

    Contenido del PDF:
    {pdf_content}

    Pregunta: {question}

    Respuesta detallada:
    """
    response = model.invoke(prompt)
    return response

def main():
    st.set_page_config(page_title="Chatbot PDF con Gemini", page_icon="📚")

    st.title("Chatbot PDF con Gemini 📚")
    st.write("Sube un PDF y haz preguntas sobre su contenido.")

    pdf_doc = st.file_uploader("Sube tu archivo PDF", type="pdf")
    
    if pdf_doc is not None:
        with st.spinner("Procesando el PDF..."):
            pdf_content = get_pdf_text(pdf_doc)
            st.success("PDF procesado con éxito. Ahora puedes hacer preguntas sobre su contenido.")
            st.session_state.pdf_content = pdf_content

    if 'pdf_content' in st.session_state:
        user_question = st.text_input("Haz una pregunta sobre el PDF:")
        if user_question:
            with st.spinner("Generando respuesta..."):
                response = get_gemini_response(user_question, st.session_state.pdf_content)
                st.write("Respuesta:")
                st.write(response)

if __name__ == "__main__":
    main()