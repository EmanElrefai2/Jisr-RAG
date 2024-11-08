import pytest
from retriever.indexing import IndexDocument
import numpy as np
import os
import faiss
from helpers.config import config
import pickle
import torch
import shutil

@pytest.fixture
def setup_test_env():
    """Setup test environment and cleanup after"""
    # Create test output directory structure
    os.makedirs("output/test_doc", exist_ok=True)
    os.makedirs("output/test_doc/vector_db", exist_ok=True)
    
    yield
    
    # Cleanup after tests
    if os.path.exists("output/test_doc"):
        shutil.rmtree("output/test_doc")

def test_load_model(embed_model):
    """Test embedding model loading"""
    assert embed_model is not None
    assert hasattr(embed_model, 'encode')

def test_create_vector_db(embed_model, mock_document, setup_test_env):
    """Test vector database creation"""
    # Create test document embeddings
    dim = config.DIM or 1024  # Use config dimension or default
    embeddings = [np.random.rand(dim).astype(np.float32)]
    output_path = "output/test_doc/vector_db/db.pickle"
    
    try:
        # Create vector database
        faiss_index = IndexDocument.create_vector_db(embeddings, output_path)
        
        # Verify index creation
        assert isinstance(faiss_index, faiss.IndexFlatIP)
        assert faiss_index.d == dim  # Verify dimension
        assert faiss_index.ntotal == len(embeddings)  # Verify number of vectors
        
        # Verify file was saved
        assert os.path.exists(output_path)
        
        # Try loading the saved index
        with open(output_path, 'rb') as f:
            loaded_index = pickle.load(f)
        assert isinstance(loaded_index, faiss.IndexFlatIP)
        
    except Exception as e:
        pytest.fail(f"Vector DB creation failed: {str(e)}")

def test_search(embed_model, mock_document, setup_test_env):
    """Test document search functionality"""
    try:
        # Setup test data
        config.EXIST_FLAG_DOC_ID = "test_doc"
        config.CHUNKS_QUEUE = mock_document
        
        # Create test vector DB
        dim = config.DIM or 1024
        test_embedding = np.random.rand(dim).astype(np.float32)
        output_path = "output/test_doc/vector_db/db.pickle"
        config.faiss_index = IndexDocument.create_vector_db([test_embedding], output_path)
        
        # Add test document to docs_mapping
        config.docs_mapping = [mock_document]
        
        # Perform search
        results = IndexDocument.search(
            query="test query",
            model=embed_model,
            faiss_index=config.faiss_index,
            docs_mapping=config.docs_mapping
        )
        
        # Verify results
        assert isinstance(results, list)
        if results:
            assert "page_content" in results[0]
            assert "meta_data" in results[0]
            assert "confidence" in results[0]
            
    except Exception as e:
        pytest.fail(f"Search test failed: {str(e)}")
        
    finally:
        # Reset config
        config.EXIST_FLAG_DOC_ID = None
        config.faiss_index = None
        config.docs_mapping = None

@pytest.fixture
def mock_document():
    return {
        "page_content": "This is a test document content.",
        "meta_data": {
            "Document_id": "test_doc_001",
            "Section Title": "Test Section",
            "page number": 1,
            "chunk_id": 0
        }
    }