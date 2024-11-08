import pytest
from llm.llm import LLM

def test_initialize_model(llm_model):
    """Test LLM model initialization"""
    model, tokenizer = llm_model
    assert model is not None
    assert tokenizer is not None

def test_generate_response(llm_model):
    """Test response generation"""
    model, tokenizer = llm_model
    test_documents = [{
        "page_content": "This is a test document.",
        "meta_data": {
            "Document_id": "test_doc",
            "chunk_id": 0
        }
    }]
    
    response, references, doc_id = LLM.generate_response(
        model=model,
        tokenizer=tokenizer,
        mood="rag",
        query="What is this document about?",
        documents=test_documents
    )
    
    assert isinstance(response, str)
    assert isinstance(references, list)
    assert isinstance(doc_id, str)