import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

import google.generativeai as genai

# Load environment variables (make sure you have GOOGLE_API_KEY in .env file)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from the specific PDF
def get_pdf_text():
    pdf_path = "Final handbook with links.pdf"  # Your PDF file
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        # Extract text
        text += page.extract_text()
        
    
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create vector store
def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("rulebook_vector_store")

def main():
    raw_text = get_pdf_text()
    text_chunks = get_text_chunks(raw_text)
    create_vector_store(text_chunks)

if __name__ == "__main__":
    main()
