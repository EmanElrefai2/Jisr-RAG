import time
import pandas as pd
import os
import json
import uuid
from PIL import Image
from typing import Tuple, Any
from surya.ocr import run_ocr
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.postprocessing.heatmap import draw_polys_on_image
from helpers.utils import (
    is_bbox_inside,
    create_folder_structure,
    block_surround,
    save_chunk,
)
import queue
import torch
from surya.model.detection.model import load_model, load_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import (
    load_processor as load_recognition_processor,
)
from surya.settings import settings
import fitz
from PIL import ImageDraw, ImageFont
from helpers.config import config

def load_models(shared_dict: dict = {}) -> dict:
    """
    Load the OCR models and processors into the shared dictionary.

    Parameters:
    shared_dict (dict): A dictionary to store the loaded models and processors.

    Returns:
    dict: The updated shared dictionary with the loaded models and processors.
    """
    # Load the OCR models and processors into the shared dictionary
    shared_dict["det_model"] = load_model()
    shared_dict["det_processor"] = load_processor()
    shared_dict["rec_model"] = load_recognition_model()
    shared_dict["rec_model"].decoder.model = torch.compile(shared_dict["rec_model"].decoder.model)
    shared_dict["rec_processor"] = load_recognition_processor()
    shared_dict["layout_model"] = load_model(
        checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
    )
    shared_dict["layout_processor"] = load_processor(
        checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
    )
    print("All Models loaded and ready.")
    return shared_dict


class OCR:
    def __init__(self, shared_dict):
        """
        Optical Character Recognition (OCR) class for extracting text from images.
        Args:
            shared_dict (dict): A dictionary containing shared models and processors.
        Attributes:
            settings (object): An object containing OCR settings.
            langs (list): A list of languages supported for text recognition.
            shared_dict (dict): A dictionary containing shared models and processors.
            text_recognition (list): A list to store the text recognition results.
            layout_detection (list): A list to store the layout detection results.
        Methods:
            predict(image: Image.Image) -> Tuple[list, list]:
                Predicts the text recognition and layout detection results for a given image.
            initialize_chunk(page_num: int, doc_id: str, chunk_id: int) -> Tuple[dict, Any, int]:
                Initializes a new chunk with default values.
            finalize_chunk(current_chunk: dict, previous_section_title: str, chunk_bbox: Any, content_length: int) -> dict:
                Finalizes the current chunk by updating the metadata.
            update_chunk_bbox(chunk_bbox: Any, text_box: dict) -> Any:
                Updates the chunk bounding box based on the text box coordinates.
            sort_boxes(text_recognition: list, layout_detection: list) -> Tuple[list, list]:
                Sorts the text boxes and layout boxes based on their y-coordinate.
            extract_page(pdf_path: str, shared_dict: dict, chunk_size: int = 200, queue: list = [], output_path: str = "/root/ODP-RAG/output") -> list:
                Extracts the text from each page of a PDF document and chunks it based on the specified chunk size.
        """
        self.settings = settings
        self.settings.IMAGE_DPI = 96
        self.settings.DETECTOR_BATCH_SIZE = 32
        self.settings.RECOGNITION_BATCH_SIZE = 32
        self.settings.RECOGNITION_STATIC_CACHE = True
        self.langs = ["ar"]
        self.shared_dict = shared_dict
        self.text_recognition = []
        self.layout_detection = []

    def predict(self, image: Image.Image) -> Tuple[list, list]:
        """
        Predicts the text recognition and layout detection results for a given image.
        Args:
            image (PIL.Image.Image): The input image.
        Returns:
            Tuple[list, list]: A tuple containing the text recognition and layout detection results.
        """
        det_model = self.shared_dict["det_model"]
        det_processor = self.shared_dict["det_processor"]
        rec_model = self.shared_dict["rec_model"]
        rec_processor = self.shared_dict["rec_processor"]
        layout_model = self.shared_dict["layout_model"]
        layout_processor = self.shared_dict["layout_processor"]
        det_prediction = batch_text_detection([image], det_model, det_processor)[0]
        layout_detection = batch_layout_detection(
            [image], layout_model, layout_processor, [det_prediction]
        )[0]

        text_recognition = run_ocr(
            [image], [self.langs], det_model, det_processor, rec_model, rec_processor
        )[0]
        return text_recognition, layout_detection

    def initialize_chunk(
        self, page_num: int, doc_id: str, chunk_id: int
    ) -> Tuple[dict, Any, int]:
        """
        Initializes a new chunk with default values.
        Args:
            page_num (int): The page number.
            doc_id (str): The document ID.
            chunk_id (int): The chunk ID.
        Returns:
            Tuple[dict, Any, int]: A tuple containing the current chunk, chunk bounding box, and content length.
        """
        ...
        current_chunk = {
            "page_content": "",
            "meta_data": {
                "Document_id": doc_id,
                "Section Title": None,
                "page number": page_num,
                "chunk_id": chunk_id,
                "chunk_bbox": None,
                "content_length": 0,
                "chunk_uuid": 0,
            },
        }
        chunk_bbox = None
        content_length = 0
        return current_chunk, chunk_bbox, content_length

    def finalize_chunk(
        self,
        current_chunk: dict,
        previous_section_title: str,
        chunk_bbox: Any,
        content_length: int,
        doc_id: str,
    ) -> dict:
        """
        Finalizes the current chunk by updating the metadata.
        Args:
            current_chunk (dict): The current chunk.
            previous_section_title (str): The previous section title.
            chunk_bbox (Any): The chunk bounding box.
            content_length (int): The content length.
        Returns:
            dict: The finalized current chunk.
        """
        temp_chunk = current_chunk.copy()
        del temp_chunk["meta_data"]["Document_id"]
        uuid_chunk = str(uuid.uuid3(uuid.NAMESPACE_URL, str(temp_chunk)))
        # delete document_id from the meta_data
        current_chunk["meta_data"]["Section Title"] = previous_section_title
        current_chunk["meta_data"]["chunk_bbox"] = chunk_bbox
        current_chunk["meta_data"]["content_length"] = content_length
        current_chunk["meta_data"]["chunk_uuid"] = uuid_chunk
        current_chunk["meta_data"]["Document_id"] = doc_id
        return current_chunk

    def update_chunk_bbox(self, chunk_bbox: Any, text_box: dict) -> Any:
        """
        Updates the chunk bounding box based on the text box coordinates.
        Args:
            chunk_bbox (Any): The current chunk bounding box.
            text_box (dict): The text box.
        Returns:
            Any: The updated chunk bounding box.
        """
        if chunk_bbox:
            x0 = min(chunk_bbox[0], text_box["bbox"][0])
            y0 = min(chunk_bbox[1], text_box["bbox"][1])
            x1 = max(chunk_bbox[2], text_box["bbox"][2])
            y1 = max(chunk_bbox[3], text_box["bbox"][3])
            chunk_bbox = [x0, y0, x1, y1]
        else:
            chunk_bbox = text_box["bbox"]
        return chunk_bbox

    def sort_boxes(
        self, text_recognition: list, layout_detection: list
    ) -> Tuple[list, list]:
        """
        Sorts the text boxes and layout boxes based on their y-coordinate.
        Args:
            text_recognition (list): The list of text recognition results.
            layout_detection (list): The list of layout detection results.
        Returns:
            Tuple[list, list]: A tuple containing the sorted text boxes and layout boxes.
        """
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
        return text_boxes, layout_boxes

    def check_chunk_exists(self, chunk_id: list, chunk_map: dict):
        """
        Check if the chunk exists in the chunk map.
        Args:
            chunk_id (list): The list of chunk IDs.
            chunk_map (dict): The chunk map dictionary contains {doc_id:[chunks_ids]}.
        Returns:
            bool: True if the chunk exists, False otherwise.
        """
        # check for matching chunk id in chunk map should all chunks to be the same
        for doc_id, chunks in chunk_map.items():
            if all([c in chunks for c in chunk_id]):
                config.EXIST_FLAG = True
                return doc_id

    def extract_page(
        self,
        pdf_path: str,
        chunk_size: int = 650,
        overlap_size: int = 50,
        queueuo: list = [],
        output_path: str = "output",
    ) -> list:
        """
        Extracts the text from each page of a PDF document and chunks it based on the specified chunk size.
        Args:
            pdf_path (str): The path to the PDF document.
            shared_dict (dict): A dictionary containing shared models and processors.
            chunk_size (int, optional): The maximum number of words allowed in each chunk. Defaults to 200.
            queue (list, optional): A list to store the extracted chunks. Defaults to [].
            output_path (str, optional): The output path for saving the extracted chunks. Defaults to "/root/ODP-RAG/output".
        Returns:
            list: A list of extracted chunks.
        """

        start_time = time.time()
        images = fitz.open(pdf_path)

        # load chunk map
        print("Loading Chunking Map")
        chunk_map_path = f"{output_path}/chunk_map.json"
        with open(chunk_map_path, "r") as f:
            chunk_map = json.load(f)
        print("Chunk Map Loaded")
        print("-"*10)

        doc_id = str(uuid.uuid4())
        output_path = f"{output_path}/{doc_id}"
        create_folder_structure(output_path)

        chunk_id = 0
        previous_section_title = None
        chunks_uuids = []

        for page_num, image in enumerate(images, start=1):
            image = image.get_pixmap(dpi=96)
            image = Image.frombytes("RGB", (image.width, image.height), image.samples)
            text_recognition, layout_detection = self.predict(image)
            text_boxes, layout_boxes = self.sort_boxes(
                text_recognition, layout_detection
            )

            current_chunk, chunk_bbox, content_length = self.initialize_chunk(
                page_num, doc_id, chunk_id
            )

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
                text_length = len(formatted_text)

                # Check if we need to start a new chunk
                if (
                    ((matched_label in ["Title", "Section-header", "Formula"]) and (content_length + text_length > 20))
                    or content_length + text_length > chunk_size
                ):
                    if current_chunk["page_content"]:
                        # Finalize and save current chunk
                        current_chunk = self.finalize_chunk(
                            current_chunk,
                            previous_section_title,
                            chunk_bbox,
                            content_length,
                            doc_id
                        )

                        queueuo.append(current_chunk)
                        config.CHUNKS_QUEUE.put(current_chunk)
                        chunks_uuids.append(current_chunk["meta_data"]["chunk_uuid"])
                        save_chunk(current_chunk, image, output_path, save_image=True)
                        chunk_id += 1
                        current_chunk, chunk_bbox, content_length = (
                            self.initialize_chunk(page_num, doc_id, chunk_id)
                        )

                    # Update section titles
                    if matched_label in ["Section-header", "Formula"]:
                        previous_section_title = text_box["text"]
                    elif matched_label == "Title":
                        previous_section_title = text_box["text"]

                current_chunk["page_content"] += formatted_text
                content_length += text_length

                # Update chunk_bbox
                chunk_bbox = self.update_chunk_bbox(chunk_bbox, text_box)
            # Save the last chunk for the page
            if current_chunk["page_content"]:
                current_chunk = self.finalize_chunk(
                    current_chunk, previous_section_title, chunk_bbox, content_length, doc_id
                )
                queueuo.append(current_chunk)
                config.CHUNKS_QUEUE.put(current_chunk)
                chunks_uuids.append(current_chunk["meta_data"]["chunk_uuid"])
                save_chunk(current_chunk, image, output_path, save_image=True)
                chunk_id += 1

            image_path = f"{output_path}/pages/page_{page_num}.png"
            image.save(image_path)

            # check for EXIST_FLAG
            if page_num == 3:
                print("Print page 3")
                config.EXIST_FLAG_DOC_ID = self.check_chunk_exists(
                    chunks_uuids, chunk_map
                )
                print("config.EXIST_FLAG_DOC_ID", config.EXIST_FLAG_DOC_ID)
                if config.EXIST_FLAG == True:
                    config.EMBED_START_FLAG.set()
                    print("file is already processed before")
                    print("EMBED_START_FLAG is set")
                    break
                else:
                    print("file is not found")
                    config.EMBED_START_FLAG.set()
                    print("EMBED_START_FLAG is not set")
            elif len(images)<3:
                config.EMBED_START_FLAG.set()
                print("EMBED_START_FLAG is set")

        chunk_map[doc_id] = chunks_uuids
        #save chunk map
        with open(chunk_map_path, "w") as f:
            json.dump(chunk_map, f)

        # Raise Flag to indicate OCR is done
        config.OCR_FLAG = True
        if config.OCR_FLAG and not(config.EXIST_FLAG):
            config.EXIST_FLAG_DOC_ID = doc_id
        end_time = time.time()
        print(
            f"Time taken to extract and chunk the pages: {end_time - start_time} seconds."
        )

        print("QUEUE STATE", config.CHUNKS_QUEUE.empty())
        print("config.OCR_FLAG", config.OCR_FLAG)

