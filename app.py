import streamlit as st
import os
import joblib
import numpy as np
import random
from dotenv import load_dotenv
import time

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Create local cache folder
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_resource(show_spinner=False)
def initialize_resources():
    """Initialize resources using Streamlit's cache_resource decorator to ensure single initialization."""
    # List of available API keys
    api_keys = [
        os.getenv("GOOGLE_API_KEY_1"),
        os.getenv("GOOGLE_API_KEY_2"),
        os.getenv("GOOGLE_API_KEY_3"),
        os.getenv("GOOGLE_API_KEY_4")
    ]

    # Remove any None or empty keys from the list
    api_keys = [key for key in api_keys if key]

    # If there are no valid keys, raise an error
    if not api_keys:
        raise Exception("No valid API key available")

    # Randomly shuffle API keys to ensure balanced load
    random.shuffle(api_keys)

    for key in api_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp') # or try 'gemini-1.5-flash' or 'gemini-pro'
            response = model.generate_content("test")

            # Initialize embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=key
            )

            # Load FAISS index
            faiss_index = FAISS.load_local(
                "rulebook_vector_store",
                embeddings,
                allow_dangerous_deserialization=True
            )

            return {
                "api_key": key,
                "embeddings": embeddings,
                "faiss_index": faiss_index
            }
        except Exception as e:
            print(f"API key error: {str(e)}")
            continue

    raise Exception("No valid API key available")

@st.cache_data(show_spinner=False)
def get_semantic_key(question: str, _embeddings) -> str:
    """Generate a semantic cache key using embeddings."""
    vector = _embeddings.embed_query(question)
    semantic_key = hash(tuple(np.round(vector, decimals=5)))
    return f"semantic_search_{semantic_key}"

def load_cache_from_disk(cache_name: str):
    """Load cache data using joblib if it exists."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    return None


@st.cache_resource(show_spinner=False)
def get_generative_model(_api_key):
    """Return a generative model using the cached API key."""
    genai.configure(api_key=_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-1219') # or try 'gemini-1.5-flash' or 'gemini-pro'
    return model


@st.cache_data(show_spinner=False)
def cached_similarity_search(question: str, _resources):
    """Efficient Semantic Caching for Query Similarity Search."""
    # Semantic Cache (using embeddings)
    semantic_key = get_semantic_key(question, _resources["embeddings"])
    cached_docs = load_cache_from_disk(semantic_key)
    if cached_docs is not None:
        return cached_docs  # Return if cached

    # Perform Search using FAISS index
    return _resources["faiss_index"].similarity_search(question)

def stream_response(model, prompt):
    """Streams the response from the model and displays it in Streamlit with letter-by-letter effect."""
    placeholder = st.empty()
    full_response = ""
    response_stream = model.generate_content(prompt, stream=True)
    for chunk in response_stream:
        text_chunk = chunk.text or "" # Handle cases where chunk.text might be None
        for char in text_chunk: # Iterate through each character
            full_response += char
            placeholder.markdown(full_response + "▌", unsafe_allow_html=False)
            time.sleep(0.001) # Add a small delay for the streaming effect, adjust as needed
    placeholder.markdown(full_response, unsafe_allow_html=False)

def process_question(question, resources):
    """Generate an answer with caching and streaming."""
    question_with_instructions = question + ". Provide a comprehensive and informative explanation. If there is any relevant link or reference, please provide that as well. **Do not give information which is incorrect or incomplete.** **Format the response clearly and concisely.** **Ensure the answer is accurate and addresses all aspects of the question.** **Avoid vague or generic responses.** **Strictly do not reference the source document explicitly (e.g., avoid phrases like 'the provided documentation', 'the document says' or 'text mentions' or similar meaning phrases) instead answer as if you are answering from you own knowledge, if any link or reference is given then give it enclosed in some text where when clicked, it opens the link, also show what it is for.** **Strictly Present the information naturally and independently, as if explaining from knowledge.** **Use headings or bullet points to improve readability.** **Include tables if they add clarity or structure to the answer.** **If information is limited, state 'Based on the available information...' and provide the most relevant details.**"

    docs = cached_similarity_search(question, resources)

    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt_template = """
    Answer the question using the provided context.  **Your response MUST be formatted as TEXT with Markdown elements, NOT as a Markdown code block.**

    **CRITICAL FORMATTING RULES - FOLLOW THESE EXACTLY:**

    **1.  NO CODE BLOCKS!  ABSOLUTELY DO NOT USE ANY CODE BLOCKS!**  Do not enclose *any* part of your response in triple backticks (```) or single backticks (`). Your output must be rendered directly as Markdown, not as code.

    2.  Format your answer beautifully in Markdown as follows for TEXT ELEMENTS ONLY:

        - **Headings:** Use ### for headings and only headings, not for normal text.
        - **Subheadings:** Use #### for subheadings and only subheadings, not for normal text.
        - **Normal Text:** Write normal text without any special characters (except for Markdown formatting).
        - **Details:** Use bullet points (- ) or numbered lists (1. ) for clarity.
        - **New Lines:** Use double new lines (\\n\\n) to separate paragraphs and sections.
        - **Emphasis:** Use ** for bold and _ for italics to highlight important information.

    3.  If the answer is not available in the provided context, clearly state:
        "The exact answer to this question is not available in the documentation. Here are some relevant details:"
        Then explain the relevant details in a clear and concise manner using the Markdown formatting rules above.

    Context: {context}

    Question: {question}

    Answer:
    """

    prompt = prompt_template.format(context=context_text, question=question_with_instructions)

    model = get_generative_model(resources["api_key"])
    stream_response(model, prompt)


def main():
    st.set_page_config(page_title="ChEAGPT")
    st.header("ChEAGPT")

    # Initialize resources once and store in session state
    if "resources" not in st.session_state:
        try:
            st.session_state.resources = initialize_resources()
        except Exception as e:
            st.error(f"Failed to initialize resources: {str(e)}")
            return

    user_question = st.text_input("Ask any question about institute policies:")
    if user_question:
        try:
            with st.spinner("Thinking..."):
                process_question(user_question, st.session_state.resources)
                st.write("ChEAGPT can make mistakes. It is advisable recheck the information")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
