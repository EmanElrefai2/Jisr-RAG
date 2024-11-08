import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import pytest
import torch
from PIL import Image
import numpy as np
from helpers.config import config
from ocr.ocr import load_models
from retriever.indexing import IndexDocument
from llm.llm import LLM

@pytest.fixture
def sample_pdf():
    """Fixture for sample PDF file"""
    # Create tests/data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "sample.pdf")

@pytest.fixture
def ocr_models():
    """Fixture for OCR models"""
    return load_models()

@pytest.fixture
def embed_model():
    """Fixture for embedding model"""
    return IndexDocument.load_model()

@pytest.fixture
def llm_model():
    """Fixture for LLM model and tokenizer"""
    model, tokenizer = LLM.initialize_model()
    return model, tokenizer

@pytest.fixture
def mock_document():
    """Fixture for mock document data"""
    return {
        "page_content": "This is a test document content.",
        "meta_data": {
            "Document_id": "test_doc_001",
            "Section Title": "Test Section",
            "page number": 1,
            "chunk_id": 0
        }
    }