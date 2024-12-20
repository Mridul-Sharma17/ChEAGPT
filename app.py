import streamlit as st
import os
import time
import joblib
from functools import lru_cache
from typing import List
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create local cache folder
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def save_cache_to_disk(cache_name: str, data):
    """Save cache data using joblib."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    joblib.dump(data, cache_path)

def load_cache_from_disk(cache_name: str):
    """Load cache data using joblib if it exists."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    return None

@lru_cache(maxsize=1)
def get_conversation_chain():
    """Return a cached conversation chain."""
    prompt_template = """
    Answer the question using the provided context. Ensure the response is comprehensive and beautifully formatted in Markdown as follows:
    - **Headings:** Use `###` for headings and only headings, not for normal text.
    - **Subheadings:** Use `####` for subheadings and only subheadings, not for normal text.
    - **Normal Text:** Write normal text without any special characters.
    - **Details:** Use bullet points (`- `) or numbered lists (`1. `) for clarity.
    - **New Lines:** Use double new lines (`\\n\\n`) to separate paragraphs and sections.
    - **Emphasis:** Use `**` for bold and `_` for italics to highlight important information.

    If the answer is not available in the provided context, clearly state:
    "The answer is not available in the provided context."

    Context: {context}

    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def cached_similarity_search(question: str):
    """Cache similarity search results for a question."""
    cache_key = f"similarity_search_{hash(question)}"
    cached_docs = load_cache_from_disk(cache_key)
    if cached_docs is not None:
        return cached_docs

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("rulebook_vector_store", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question)
    save_cache_to_disk(cache_key, docs)
    return docs

def process_question(question):
    """Generate an answer with caching."""
    docs = cached_similarity_search(question)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config(page_title="ChEAGPT")
    st.header("ChEAGPT")

    user_question = st.text_input("Ask a question about the ChEA Handbook:")
    if user_question:
        modified_query = user_question + ". Explain everything in detail and in a beautifully formatted way. " \
                                         "Always start with a heading best suited to the question. " \
                                         "End with this Do you need any additional help with (current question's topic)..."

        with st.spinner("Thinking..."):
            response = process_question(modified_query)
            st.write(response)
            st.write("ChEAGPT can make mistakes. Please verify the information from ChEA Handbook.")

if __name__ == "__main__":
    main()