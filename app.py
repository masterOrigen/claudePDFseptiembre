import streamlit as st
import google.generativeai as genai
import PyPDF2
import os

# Configurar la API de Gemini
genai.configure(api_key='TU_CLAVE_API_AQUI')

# Función para extraer texto de un PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Configurar la interfaz de Streamlit
st.title("Chat con PDF usando Gemini")

# Subir archivo PDF
uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Área de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Haz una pregunta sobre el PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta usando Gemini
        response = genai.generate_text(
            model="gemini-pro",
            prompt=f"Basado en el siguiente texto de un PDF: {pdf_text}\n\nPregunta: {prompt}\nRespuesta:",
            temperature=0.3,
            max_output_tokens=2048,
        )

        with st.chat_message("assistant"):
            st.markdown(response.result)
        st.session_state.messages.append({"role": "assistant", "content": response.result})
