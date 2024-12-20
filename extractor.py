import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

import google.generativeai as genai

# Load environment variables (make sure you have GOOGLE_API_KEY in .env file)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from the specific PDF
def get_pdf_text():
    pdf_path = "Chea Handbook Final.pdf"  # Your PDF file
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create vector store
def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("rulebook_vector_store")

def main():
    raw_text = get_pdf_text()
    extra_text = """
    Schedule of Summer Term (C) / Summer Courses
    The following are the key events and deadlines for the summer term, along with their respective dates:
    1. Last Date to Receive List of Summer Courses from Academic Units
    Date: 30th April 2025 (Wednesday)
    2. Display of Summer Courses List on ASC
    Date: 7th May 2025 (Wednesday)
    3. Registration
    Dates: 14th May 2025 (Wednesday) to 19th May 2025 (Monday)
    4. Instruction Begins
    Date: 20th May 2025 (Tuesday)
    5. Last Date for Course Adjustment
    Date: 27th May 2025 (Tuesday)
    6. Last Date of Instruction
    Date: 12th July 2025 (Saturday)
    7. Term-End Final Exam
    Dates: 13th July 2025 (Sunday) to 17th July 2025 (Thursday)
    8. Last Date for Showing Evaluated Answer Scripts
    Date: 19th July 2025 (Saturday)
    9. Online Submission of Grades
    Dates: 13th July 2025 (Sunday) to 20th July 2025 (Sunday)
    """
    text_chunks = get_text_chunks(raw_text)
    create_vector_store(text_chunks)

if __name__ == "__main__":
    main()
