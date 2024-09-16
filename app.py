import streamlit as st
import pdfplumber
from anthropic import Anthropic
from dotenv import load_dotenv
import os

# [El código de estilo permanece igual]

# Cargar variables de entorno
load_dotenv()

# Configurar Claude API
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    st.error("No se encontró la CLAUDE_API_KEY en las variables de entorno")
    st.stop()

anthropic = Anthropic(api_key=CLAUDE_API_KEY)

def extract_text_from_pdf(pdf_file):
    # [Esta función permanece igual]

def get_claude_response(messages, system_prompt):
    # [Esta función permanece igual]

def main():
    st.markdown("<h2 style='color: black;'>PDF Insights Chat con Claude 3.5 Sonnet</h2>", unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = None
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""

    uploaded_file = st.file_uploader("Sube un archivo PDF con el que deseas interactuar", type="pdf")

    if uploaded_file is not None and st.session_state.pdf_content is None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success(f"PDF procesado exitosamente. Contenido extraído: {len(pdf_text)} caracteres.")
            st.session_state.pdf_content = pdf_text
        else:
            st.error("No se pudo procesar el PDF. Intenta con otro archivo.")

    if st.session_state.pdf_content:
        st.write("Ahora puedes hacer preguntas sobre el contenido del PDF.")
        
        # Mostrar el historial de chat
        for entry in st.session_state.chat_history:
            st.write(f"Pregunta: {entry['question']}")
            st.write(f"Respuesta: {entry['answer']}")
            st.markdown("---")

        # Área de texto para la nueva pregunta
        user_question = st.text_area("Escribe tu pregunta aquí:", value=st.session_state.user_question, height=100, key="question_input")
        
        if st.button("Enviar pregunta"):
            if user_question:
                with st.spinner("Claude está pensando..."):
                    system_prompt = f"Eres un asistente útil que responde preguntas basadas en el siguiente contenido de un documento PDF:\n\n{st.session_state.pdf_content[:8000]}"
                    
                    messages = [
                        {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
                        for i, msg in enumerate(sum([(entry['question'], entry['answer']) for entry in st.session_state.chat_history], ()))
                    ]
                    messages.append({"role": "user", "content": user_question})
                    
                    response = get_claude_response(messages, system_prompt)
                    
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response
                    })
                    
                    # Limpiar el área de texto
                    st.session_state.user_question = ""
                
                st.experimental_rerun()

if __name__ == "__main__":
    main()
