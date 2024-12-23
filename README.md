# ChEAGPT - Your AI Assistant for ChEA Handbook

## Overview
ChEAGPT is an AI-powered application that helps users interact with the ChEA Handbook content through natural language queries. It uses advanced language models and vector store technology to provide accurate, context-aware responses.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- ChEA Handbook PDF file
- Multiple Google API keys for redundancy

### Virtual Environment Setup
```bash
python -m venv cheagpt_env
source cheagpt_env/bin/activate  # Linux/MacOS
# or
cheagpt_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file with multiple API keys:
```
GOOGLE_API_KEY_1=your_first_key
GOOGLE_API_KEY_2=your_second_key
GOOGLE_API_KEY_3=your_third_key
GOOGLE_API_KEY_4=your_fourth_key
```

## Components

### 1. Extractor (extractor.py)
- Extracts text from PDF
- Splits text into manageable chunks
- Creates FAISS vector store using Google's embeddings

### 2. Cache Generator (cache_generator.py)
- Implements semantic caching system
- Pre-generates cache for common queries
- Optimizes response time
- Handles multiple API keys

### 3. Main Application (app.py)
- Streamlit-based user interface
- Implements question-answering system
- Features multiple API key failover
- Markdown formatting for responses

## Usage

1. Generate vector store:
```bash
python extractor.py
```

2. Generate cache (optional):
```bash
python cache_generator.py
```

3. Launch application:
```bash
streamlit run app.py
```

## Features
- Robust API key management
- Semantic caching system
- Markdown-formatted responses
- Multiple failover mechanisms
- Pre-generated cache for common queries

## Performance Optimizations
- Multi-level caching (memory and disk)
- API key redundancy
- Efficient vector similarity search
- Response streaming

## Important Notes
- Always verify information with official ChEA Handbook
- System uses multiple API keys for reliability
- Responses are cached for better performance

## License
ChEAGPT is owned by Mridul Sharma

## Contributing
Feel free to submit issues and enhancement requests.