import streamlit as st
import google.generativeai as genai
import PyPDF2
import os

# Configurar la API de Gemini usando una variable de entorno
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("La clave API de Gemini no está configurada. Por favor, configura la variable de entorno GEMINI_API_KEY.")
    st.stop()

genai.configure(api_key=api_key)

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
        response = genai.generate_content(
            model="models/gemini-pro",
            contents=f"Basado en el siguiente texto de un PDF: {pdf_text}\n\nPregunta: {prompt}\nRespuesta:",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048,
            )
        )

        with st.chat_message("assistant"):
            st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
