# Project: Jisr RAG


## Overview
A chatbot capable of answering users’ questions based on the knowledge extracted from PDF documents. The application should allow users to upload documents and chat with the chatbot using a simple, accessible interface.


## Python Version

   > python 3.10


***

## Installation

1. Clone the repository: `git clone https://github.com/EmanElrefai2/Jisr-RAG`
2. Navigate to the project directory: `cd Jisr-RAG`
3. Create the environment: `python3.10 -m venv myenv`
4. Activate the environment: `source myenv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Run the main application: `python main.py`


## Usage

1. Run the main application: `python main.py`
2. Access the web interface at `http://localhost:8082`


## Docker

To run the application using Docker:

1. Build the Docker image: `docker build -t gradio-app .`
2. Run the Docker container: `docker run -p 8082:8082 gradio-app`
3. Access the application using the appropriate IP address or domain name and port 8082.

## Testing

1. Run tests: `pytest tests/`

## Presentation
- [Slides](https://docs.google.com/presentation/d/1VRMtCKVCx3yLVEoGoagUFLNBOIeJ0o-mouakylgRlVE/edit?usp=sharing)
- [Video Demo](https://drive.google.com/file/d/16EZTPmdL0qQ7RSEDc-yLkM5dO2gFlCjI/view?usp=sharing)
## Documentation

### Features

- PDF document upload and processing
- OCR text extraction with layout analysis
- Semantic search for relevant document sections
- Question answering using Qwen2.5 language model
- Web interface using Gradio

### API Documentation

### REST Endpoints

##### POST /api/upload
Upload and process a PDF document
- Request: Multipart form data with PDF file
- Response: Processing status message

##### POST /api/chat
Send a question to the chatbot
- Request: JSON with query and optional mood
- Response: Bot response with citations

### Architecture

The application consists of several key components:

ODP-RAG/
├── app.py
├── requirements.txt
├── setup.py           # New file for package configuration
├── helpers/
│   ├── __init__.py
│   ├── config.py
│   └── utils.py
├── ocr/
│   ├── __init__.py
│   └── ocr.py
├── retriever/
│   ├── __init__.py
│   └── indexing.py
├── llm/
│   ├── __init__.py
│   └── llm.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_ocr.py
    ├── test_indexing.py
    ├── test_llm.py
    └── data/
        └── sample.pdf

1. OCR Module
- Uses Surya OCR for text extraction
- Performs layout analysis
- Chunks documents into manageable sections

2. Retriever Module
- Uses sentence transformers for document embedding
- Implements semantic search using FAISS
- Maintains document index

3. LLM Module
- Uses Qwen2.5 for text generation
- Implements RAG for accurate responses
- Provides citation support


4. Unit Tests
- OCR module testing
- Indexing functionality
- LLM response generation
- Individual component validation

### Usage

1. Upload a PDF document through the web interface or API
2. Wait for processing completion
3. Ask questions about the document content
4. View responses with relevant citations

### Development

The project follows a modular architecture with clear separation of concerns:
- `ocr/`: Document processing and text extraction
- `retriever/`: Document indexing and search
- `llm/`: Language model integration
- `helpers/`: Utility functions




