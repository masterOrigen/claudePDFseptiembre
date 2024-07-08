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
    st.success(f"Vector store created with {len(chunks)} chunks.")

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant tasked with answering questions based on the provided context. 
    Use the following pieces of context to answer the user's question. 
    If the answer is not in the context, say "I don't have enough information to answer that question."

    Context: {context}

    Human: {question}
    AI Assistant: Let me analyze the context and provide an answer to your question.
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   max_output_tokens=2048,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001")  # type: ignore

        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=4)
        
        st.write(f"Found {len(docs)} relevant documents.")
        
        context = "\n".join([doc.page_content for doc in docs])
        st.write(f"Context length: {len(context)} characters")
        
        chain = get_conversational_chain()

        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        return response["output_text"]
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your question."

st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader(
        "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            st.write(f"Extracted {len(raw_text)} characters of text.")
            text_chunks = get_text_chunks(raw_text)
            st.write(f"Created {len(text_chunks)} text chunks.")
            get_vector_store(text_chunks)

st.title("Chat with PDF files using Gemini🤖")
st.write("Welcome to the chat!")
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = user_input(prompt)
            st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})