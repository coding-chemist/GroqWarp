import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama.llms import OllamaLLM as Ollama
import pandas as pd
import altair as alt
import time

from dotenv import load_dotenv

load_dotenv()

# Load the GROQ & Google API Key from env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.title("RAG with Groq")

# Load LLMs
llm_groq = ChatGroq(groq_api_key=GROQ_API_KEY,
                    model_name="gemma2-9b-it")

llm_ollama = Ollama(model="gemma2:9b")

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context only. 
        Provide the most accurate response based on the question.
        Do not add any new information
        <context>
        {context}
        </context>
        Question: {input} 
        """
    )

# Load FAISS Vector Store
VECTOR_STORE_PATH = "./faiss_index"

@st.cache_resource
def load_vector_store():
    """Loads or creates the FAISS vector store efficiently."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(VECTOR_STORE_PATH):
        st.write("✅ Loaded existing FAISS Vector Store")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        st.write("⏳ Creating new FAISS Vector Store...")
        loader = PyPDFDirectoryLoader("./pdfs")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(docs)

        vectors = FAISS.from_documents(final_docs, embeddings)
        vectors.save_local(VECTOR_STORE_PATH)  # Save FAISS index
        st.write("✅ Vector Store Created & Saved!")
        return vectors


if "vectors" not in st.session_state:
    st.session_state.vectors = load_vector_store()

# Ask a question
prompt1 = st.text_input("What you want to ask?")


if prompt1:
    document_chain_groq = create_stuff_documents_chain(llm_groq, prompt)
    document_chain_ollama = create_stuff_documents_chain(llm_ollama, prompt)

    retriever = st.session_state.vectors.as_retriever()

    retriever_chain_groq = create_retrieval_chain(retriever, document_chain_groq)
    retriever_chain_ollama = create_retrieval_chain(retriever, document_chain_ollama)

    # Display results
    st.subheader("Results:")

    groq_col, ollama_col = st.columns(2)
    with groq_col:
        st.write("### Groq (Gemma 2-9B-IT):")

        # Measure time for Groq
        start_groq = time.time()
        response_groq = retriever_chain_groq.invoke({'input': prompt1})
        end_groq = time.time()
        groq_time = round(end_groq - start_groq, 3)
    
        st.write(response_groq['answer'])
        st.write(f"⏳ Time Taken: {groq_time}s")

        # Similarity Search Context
        with st.expander("Similarity Search (Groq)"):
            for i, doc in enumerate(response_groq['context']):
                st.write(doc.page_content)
                st.write("- " * 40)

    with ollama_col:
        st.write("### Ollama (Gemma 2-9B):")

        # Measure time for Ollama
        start_ollama = time.time()
        response_ollama = retriever_chain_ollama.invoke({'input': prompt1})
        end_ollama = time.time()
        ollama_time = round(end_ollama - start_ollama, 3)

        st.write(response_ollama['answer'])
        st.write(f"⏳ Time Taken: {ollama_time}s")

        # Similarity Search Context
        with st.expander("Similarity Search (Ollama)"):
            for i, doc in enumerate(response_ollama['context']):
                st.write(doc.page_content)
                st.write("- " * 40) 

    # Compare times
    st.write("### 🆚 Performance Comparison:")
    # Plot the time taken using bar chart
    data = pd.DataFrame({"Model": ["Groq", "Ollama"], "Time": [groq_time, ollama_time]})

    # Create bar chart with customization
    chart = (
        alt.Chart(data)
        .mark_bar(color="#FF9B85", size=60)  # Adjust bar width
        .encode(
            x=alt.X("Model", axis=alt.Axis(labelAngle=0)),  # Rotate x labels
            y=alt.Y("Time", axis=alt.Axis(grid=False), scale=alt.Scale(zero=True)),  # Remove grid lines
            tooltip=["Model", "Time"],  # Customize tooltip
        )
    )

    # Display chart in Streamlit
    st.altair_chart(chart)

    if groq_time < ollama_time:
        st.write(f"✅ **Groq was faster by {round(ollama_time - groq_time, 3)} seconds**")
    else:
        st.write(f"✅ **Ollama was faster by {round(groq_time - ollama_time, 3)} seconds**")
