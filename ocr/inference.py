import time
import os
import uuid
from PIL import Image
from surya.ocr import run_ocr
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from helpers.utils import (
    is_bbox_inside,
    create_folder_structure,
    block_surround,
    save_chunk
)
from surya.model.detection.model import load_model, load_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import (
    load_processor as load_recognition_processor,
)
from surya.settings import settings
import fitz

settings.IMAGE_DPI = 96
settings.DETECTOR_BATCH_SIZE = 32
settings.RECOGNITION_BATCH_SIZE = 32
langs = ["ar"]


def load_models(shared_dict):
    shared_dict["det_model"] = load_model()
    shared_dict["det_processor"] = load_processor()
    shared_dict["rec_model"] = load_recognition_model()
    shared_dict["rec_processor"] = load_recognition_processor()
    shared_dict["layout_model"] = load_model(
        checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
    )
    shared_dict["layout_processor"] = load_processor(
        checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
    )
    print("All Models loaded and ready.")
    return shared_dict


def OCR(image: Image.Image, shared_dict):
    det_model = shared_dict["det_model"]
    det_processor = shared_dict["det_processor"]
    rec_model = shared_dict["rec_model"]
    rec_processor = shared_dict["rec_processor"]

    text_recognition = run_ocr(
        [image], [langs], det_model, det_processor, rec_model, rec_processor
    )[0]
    return text_recognition


def Layout(image: Image.Image, shared_dict):
    layout_model = shared_dict["layout_model"]
    layout_processor = shared_dict["layout_processor"]
    det_model = shared_dict["det_model"]
    det_processor = shared_dict["det_processor"]

    det_pred = batch_text_detection([image], det_model, det_processor)[0]
    layout_detection = batch_layout_detection(
        [image], layout_model, layout_processor, [det_pred]
    )[0]
    return layout_detection


def extract_page(pdf_path: str, shared_dict, chunk_size: int = 200, queue: list = [],output_path: str = "RAG_DEMO"):
    start_time = time.time()
    images = fitz.open(pdf_path)
    doc_id = str(uuid.uuid3(uuid.NAMESPACE_URL, pdf_path))
    create_folder_structure(output_path, doc_id)

    chunk_id = 0
    previous_section_title = None

    output_path = f"{output_path}/{doc_id}"
    for page_num, image in enumerate(images, start=1):
        image_path = f"{output_path}/pages/page_{page_num}.png"
        image = image.get_pixmap(dpi=96)
        image.save(image_path)
        image = Image.open(image_path)
        start_time_chunking = time.time()
        text_recognition = OCR(image, shared_dict)
        layout_detection = Layout(image, shared_dict)

        layout_boxes = sorted(
            [{"bbox": p.bbox, "label": p.label} for p in layout_detection.bboxes],
            key=lambda x: x["bbox"][1],
        )
        text_boxes = sorted(
            [
                {"bbox": t.bbox, "text": t.text, "confidence": t.confidence}
                for t in text_recognition.text_lines
            ],
            key=lambda x: x["bbox"][1],
        )
        start_time_chunking = time.time()
        current_chunk = {
            "page_content": "",
            "meta_data": {
                "Document_id": doc_id,
                "Section Title": None,
                "page number": page_num,
                "chunk_id": chunk_id,
                "chunk_bbox": None,
                "content_length": 0,
                "chunking_time": 0,
                "chunk_uuid": 0
            },
        }
        chunk_bbox = None
        content_length = 0

        for text_box in text_boxes:
            matched_label = None
            for layout_box in layout_boxes:
                if is_bbox_inside(layout_box["bbox"], text_box["bbox"]):
                    matched_label = layout_box["label"]
                    if matched_label in ["Title", "Section-header", "Formula"]:
                        if len(text_box["text"]) < 4:
                            matched_label = "Text"
                    break
            if not matched_label:
                matched_label = "Text"

            formatted_text = block_surround(text_box["text"], matched_label)
            text_length = len(formatted_text.split())

            # Check if we need to start a new chunk
            if (
                matched_label in ["Title", "Section-header", "Formula"]
                or content_length + text_length > chunk_size
            ):
                if current_chunk["page_content"]:
                    # Finalize and save current chunk
                    current_chunk["meta_data"]["Section Title"] = previous_section_title
                    current_chunk["meta_data"]["chunk_bbox"] = chunk_bbox
                    current_chunk["meta_data"]["content_length"] = content_length
                    current_chunk["meta_data"]["chunking_time"] = time.time() - start_time_chunking
                    current_chunk['meta_data']['chunk_uuid'] = str(uuid.uuid3(uuid.NAMESPACE_URL, str(current_chunk)))
                    queue.append(current_chunk)
                    save_chunk(current_chunk, image, output_path, save_image = True)
                    chunk_id += 1
                    start_time_chunking = time.time()
                    current_chunk = {
                        "page_content": "",
                        "meta_data": {
                            "Document_id": doc_id,
                            "Section Title": None,
                            "page number": page_num,
                            "chunk_id": chunk_id,
                            "chunk_bbox": None,
                            "content_length": 0,
                            "chunking_time": 0
                        },
                    }
                    chunk_bbox = None
                    content_length = 0

                # Update section titles
                if matched_label in ["Section-header", "Formula"]:
                    previous_section_title = text_box["text"]
                elif matched_label == "Title":
                    previous_section_title = text_box["text"]

            current_chunk["page_content"] += formatted_text
            content_length += text_length

            # Update chunk_bbox
            if chunk_bbox:
                x0 = min(chunk_bbox[0], text_box["bbox"][0])
                y0 = min(chunk_bbox[1], text_box["bbox"][1])
                x1 = max(chunk_bbox[2], text_box["bbox"][2])
                y1 = max(chunk_bbox[3], text_box["bbox"][3])
                chunk_bbox = [x0, y0, x1, y1]
            else:
                chunk_bbox = text_box["bbox"]

        # Save the last chunk for the page
        if current_chunk["page_content"]:
            current_chunk["meta_data"]["Section Title"] = previous_section_title
            current_chunk["meta_data"]["chunk_bbox"] = chunk_bbox
            current_chunk["meta_data"]["content_length"] = content_length
            current_chunk['meta_data']['chunking_time'] = time.time() - start_time_chunking
            current_chunk['meta_data']['chunk_uuid'] = str(uuid.uuid3(uuid.NAMESPACE_URL, str(current_chunk)))
            queue.append(current_chunk)
            save_chunk(current_chunk, image, output_path, save_image=True)
            chunk_id += 1
        
    end_time = time.time()
    print(
        f"Time taken to extract and chunk the pages: {end_time - start_time} seconds."
    )
    return queue
