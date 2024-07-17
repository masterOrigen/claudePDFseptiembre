import streamlit as st
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Configurar el estilo
st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
    }
    .st-emotion-cache-10trblm {
        color: black;
    }
    /* Estilo para todos los textos */
    .stApp p, .stApp label, .stApp div {
        color: black;
    }
    /* Estilo específico para el botón de carga de archivos */
    .stFileUploader label {
        color: black !important;
    }
    /* Estilo para los mensajes de éxito y error */
    .stSuccess, .stError {
        color: black !important;
    }
    /* Estilo para el botón "Enviar pregunta" */
    .stButton > button {
        color: #ffffff !important;
        background-color: #d72529 !important;
        border-color: #d72529 !important;
    }
    .stButton > button:hover {
        color: white !important;
        background-color: #000000 !important;
        border-color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=2000,
        ))
        return response.text
    except Exception as e:
        return f"Error al generar respuesta: {str(e)}"

def main():
    st.markdown("<h2 style='color: black;'>PDF Insights Chat</h2>", unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Sube un archivo PDF con el que deseas interactuar", type="pdf")

    if uploaded_file is not None:
        if 'pdf_content' not in st.session_state:
            pdf_text = extract_text_from_pdf(uploaded_file)
            if pdf_text:
                st.success(f"PDF procesado exitosamente. Contenido extraído: {len(pdf_text)} caracteres.")
                st.session_state.pdf_content = pdf_text
            else:
                st.error("No se pudo procesar el PDF. Intenta con otro archivo.")

    if 'pdf_content' in st.session_state:
        st.write("Ahora puedes hacer preguntas sobre el contenido del PDF.")
        
        # Mostrar el historial de chat
        for entry in st.session_state.chat_history:
            st.write(f"Pregunta: {entry['question']}")
            st.write(f"Respuesta: {entry['answer']}")
            st.markdown("---")

        # Área de texto para la nueva pregunta
        user_question = st.text_area("Escribe tu pregunta aquí:", height=100)
        if st.button("Enviar pregunta"):
            if user_question:
                prompt = f"""
                Basándote en el siguiente contenido de un documento PDF, responde la pregunta de manera detallada y extensa. 
                Utiliza toda la información relevante del documento para proporcionar una respuesta completa y exhaustiva.
                Si la información no está directamente en el contenido, intenta inferir o relacionar conceptos basándote en lo que sabes.
                Si realmente no hay información suficiente, indica que no puedes encontrar esa información específica en el documento proporcionado.

                Contenido del documento PDF:
                {st.session_state.pdf_content[:8000]}

                Pregunta: {user_question}

                Respuesta detallada:
                """

                response = get_gemini_response(prompt)
                
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": response
                })
                
                st.experimental_rerun()

if __name__ == "__main__":
    main()
