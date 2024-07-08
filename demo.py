import streamlit as st
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Configurar Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("No se encontró la GOOGLE_API_KEY en las variables de entorno")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error al extraer texto del PDF: {e}")
        return None

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error al generar respuesta: {str(e)}"

def main():
    st.title("Chatbot PDF con Gemini")

    uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success(f"PDF procesado. Contenido extraído: {len(pdf_text)} caracteres.")
            st.session_state.pdf_content = pdf_text
            
            # Mostrar los primeros 500 caracteres del contenido del PDF
            st.text_area("Primeros 500 caracteres del PDF:", pdf_text[:500], height=200)
        else:
            st.error("No se pudo procesar el PDF. Intenta con otro archivo.")

    if 'pdf_content' in st.session_state:
        user_question = st.text_input("Haz una pregunta sobre el PDF:")
        if user_question:
            prompt = f"""
            Contenido del documento PDF:
            {st.session_state.pdf_content[:4000]}

            Pregunta: {user_question}

            Por favor, responde la pregunta basándote únicamente en la información proporcionada en el contenido del documento.
            Si la información no está disponible, indica que no puedes encontrar esa información en el documento proporcionado.
            """

            response = get_gemini_response(prompt)
            
            st.write("Respuesta:")
            st.write(response)

if __name__ == "__main__":
    main()