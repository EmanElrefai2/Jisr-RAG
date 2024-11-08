import pytest
from ocr.ocr import OCR
from PIL import Image
import os
import queue
import threading
from helpers.config import config
import shutil
import fitz
import tempfile

@pytest.fixture
def setup_test_env():
    """Setup test environment with necessary directories and files"""
    # Create test directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Create a test PDF file
    pdf_path = create_test_pdf()
    
    # Setup config
    config.CHUNKS_QUEUE = queue.Queue()
    config.EXIST_FLAG = False
    config.OCR_FLAG = False
    config.EMBED_START_FLAG = threading.Event()
    
    yield pdf_path
    
    # Cleanup
    if os.path.exists("output"):
        shutil.rmtree("output")
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

def create_test_pdf():
    """Create a simple test PDF file"""
    pdf_path = "tests/data/sample.pdf"
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    # Create a PDF with some text
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is a test document.")
    doc.save(pdf_path)
    doc.close()
    
    return pdf_path

def test_ocr_initialization(ocr_models):
    """Test OCR class initialization"""
    ocr = OCR(ocr_models)
    assert ocr.settings is not None
    assert ocr.langs == ["ar"]
    assert ocr.shared_dict is not None

def test_predict_method(ocr_models):
    """Test OCR predict method"""
    ocr = OCR(ocr_models)
    # Create a simple test image with text
    img = Image.new('RGB', (200, 100), color='white')
    text_recognition, layout_detection = ocr.predict(img)
    assert text_recognition is not None
    assert layout_detection is not None

def test_initialize_chunk(ocr_models):
    """Test chunk initialization"""
    ocr = OCR(ocr_models)
    current_chunk, chunk_bbox, content_length = ocr.initialize_chunk(
        page_num=1,
        doc_id="test_doc",
        chunk_id=0
    )
    assert current_chunk["meta_data"]["page number"] == 1
    assert current_chunk["meta_data"]["Document_id"] == "test_doc"
    assert current_chunk["meta_data"]["chunk_id"] == 0
    assert chunk_bbox is None
    assert content_length == 0

def test_update_chunk_bbox(ocr_models):
    """Test chunk bbox update"""
    ocr = OCR(ocr_models)
    chunk_bbox = None
    text_box = {"bbox": [10, 10, 100, 50]}
    
    # First update with no existing bbox
    updated_bbox = ocr.update_chunk_bbox(chunk_bbox, text_box)
    assert updated_bbox == [10, 10, 100, 50]
    
    # Update with existing bbox
    text_box2 = {"bbox": [5, 5, 110, 60]}
    updated_bbox = ocr.update_chunk_bbox(updated_bbox, text_box2)
    assert updated_bbox == [5, 5, 110, 60]

def test_extract_page(ocr_models, setup_test_env):
    """Test PDF extraction"""
    ocr = OCR(ocr_models)
    pdf_path = setup_test_env
    
    try:
        # Initialize chunk map
        chunk_map_path = os.path.join("output", "chunk_map.json")
        with open(chunk_map_path, "w") as f:
            f.write("{}")
        
        # Extract page
        ocr.extract_page(
            pdf_path=pdf_path,
            chunk_size=650,
            overlap_size=50
        )
        
        # Verify outputs
        assert os.path.exists("output")
        assert config.OCR_FLAG is True
        assert not config.CHUNKS_QUEUE.empty()
        
        # Verify chunks were created
        first_chunk = config.CHUNKS_QUEUE.get()
        assert isinstance(first_chunk, dict)
        assert "page_content" in first_chunk
        assert "meta_data" in first_chunk
        
    except Exception as e:
        pytest.fail(f"PDF extraction failed: {str(e)}")

def test_sort_boxes(ocr_models):
    """Test box sorting functionality"""
    ocr = OCR(ocr_models)
    text_recognition = type('obj', (object,), {
        'text_lines': [
            type('line', (object,), {'bbox': [0, 20, 100, 40], 'text': 'Second', 'confidence': 0.9}),
            type('line', (object,), {'bbox': [0, 0, 100, 20], 'text': 'First', 'confidence': 0.9})
        ]
    })
    
    layout_detection = type('obj', (object,), {
        'bboxes': [
            type('box', (object,), {'bbox': [0, 20, 100, 40], 'label': 'text'}),
            type('box', (object,), {'bbox': [0, 0, 100, 20], 'label': 'header'})
        ]
    })
    
    text_boxes, layout_boxes = ocr.sort_boxes(text_recognition, layout_detection)
    
    assert len(text_boxes) == 2
    assert len(layout_boxes) == 2
    assert text_boxes[0]['bbox'][1] < text_boxes[1]['bbox'][1]  # Check y-coordinate ordering