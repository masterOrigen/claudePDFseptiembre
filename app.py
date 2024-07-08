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

# Las funciones get_pdf_text y get_text_chunks permanecen sin cambios

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# La función clear_chat_history permanece sin cambios

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    return response

# La función main permanece sin cambios excepto por esta parte:

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = user_input(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response['output_text']:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    if response is not None:
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

if __name__ == "__main__":
    main()