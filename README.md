# ChEAGPT - Your AI Assistant for ChEA Handbook

## Overview
ChEAGPT is an AI-powered application that helps users interact with the ChEA Handbook content through natural language queries. It uses advanced language models and vector store technology to provide accurate, context-aware responses.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- ChEA Handbook PDF file
- Google API key

### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv cheagpt_env

# Activate virtual environment
## For Windows
cheagpt_env\Scripts\activate
## For Linux/MacOS
source cheagpt_env/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Environment Configuration
1. Create a `.env` file in the project root
2. Add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Project Structure
- `vector_store.py`: Creates vector embeddings from the PDF
- `app.py`: Main Streamlit application
- `Chea Handbook Final.pdf`: Source PDF document
- `rulebook_vector_store/`: Directory containing vector embeddings

## Usage

### 1. Generate Vector Store
First, run the vector store creation script:
```bash
python vector_store.py
```

### 2. Launch the Application
Start the Streamlit application:
```bash
streamlit run app.py
```

### 3. Using the Interface
- Enter your question in the text input field
- Wait for the AI to process and provide a formatted response
- Verify the information with the official ChEA Handbook

## Features
- PDF text extraction
- Text chunking and vector embeddings
- Efficient caching system
- Markdown-formatted responses
- Context-aware answers
- User-friendly interface

## Performance Optimization
- LRU cache for conversation chain
- Disk-based caching for similarity search results
- Efficient vector store implementation

## Important Note
The application should be used as a reference tool. Always verify important information with the official ChEA Handbook.

## License
ChEAGPT is owned by Mridul Sharma

## Contributing
Feel free to submit issues and enhancement requests.
