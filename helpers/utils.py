import os
from PIL import ImageDraw, ImageFont
import json
import queue
import threading
import re

from helpers.config import config


def reset_flags():
    config.CHUNKS_QUEUE = queue.Queue()
    config.EXIST_FLAG = False
    config.OCR_FLAG = False
    config.EMBED_START_FLAG = threading.Event()


def extract_references(text):
    references = re.findall(r'\[\d+\]', text)
    cleaned_text = re.sub(r'\[\d+\]', '', text)
    return references, cleaned_text


# Helper function to check if one bbox is inside another
def is_bbox_inside(outer, inner, threshold=0.2):
    """
    Check if one bbox (inner) is significantly inside another bbox (outer).
    The threshold controls the required overlap ratio.
    Parameters:
        - outer (tuple): The coordinates of the outer bbox in the format (x1, y1, x2, y2).
        - inner (tuple): The coordinates of the inner bbox in the format (x1, y1, x2, y2).
        - threshold (float): The required overlap ratio. Default is 0.2.
    Returns:
        - bool: True if the inner bbox is significantly inside the outer bbox, False otherwise.
    """
    outer_x1, outer_y1, outer_x2, outer_y2 = outer
    inner_x1, inner_y1, inner_x2, inner_y2 = inner

    # Calculate the area of intersection
    inter_x1 = max(outer_x1, inner_x1)
    inter_y1 = max(outer_y1, inner_y1)
    inter_x2 = min(outer_x2, inner_x2)
    inter_y2 = min(outer_y2, inner_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate area of the inner bbox
    inner_area = (inner_x2 - inner_x1) * (inner_y2 - inner_y1)

    # Calculate overlap ratio
    overlap_ratio = inter_area / inner_area if inner_area > 0 else 0

    return overlap_ratio >= threshold


# Helper function to create folder structure
def create_folder_structure(output_path):
    """
    Create the folder structure for the output files.

    Parameters:
        path (str): The base path where the folder structure will be created.

    Returns:
        None
    """
    # check if the output path exists
    if not os.path.exists(output_path):
        # create the output path
        print(f"Creating output folder: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    print(f"Creating folder needed in {output_path}")
    os.makedirs(f"{output_path}/pages", exist_ok=True)
    os.makedirs(f"{output_path}/chunks", exist_ok=True)
    os.makedirs(f"{output_path}/pages_layout", exist_ok=True)
    os.makedirs(f"{output_path}/images", exist_ok=True)
    os.makedirs(f"{output_path}/texts", exist_ok=True)
    os.makedirs(f"{output_path}/markdowns", exist_ok=True)
    os.makedirs(f"{output_path}/chunks_layout", exist_ok=True)
    os.makedirs(f"{output_path}/vector_db", exist_ok=True)


# Function to structure markdown text based on detected block type
def block_surround(text, block_type):
    """
    Surrounds the given text with specific formatting based on the block type.
    Parameters:
        - text (str): The text to be surrounded.
        - block_type (str): The type of block to determine the formatting.
    Returns:
        - str: The formatted text based on the block type.
    """

    text = text.strip()  # Clean up any leading/trailing whitespace
    
    # Format based on block type
    if block_type == "Section-header":
        return f"\n## {text}\n" if len(text) > 6 else f"{text}\n"
    
    elif block_type == "Title":
        return f"# {text}\n" if len(text) > 6 else f"{text}\n"
    
    elif block_type == "Table":
        return f"{text}\t\t"  # Ensure table text has tab spacing
    
    elif block_type == "Figure":
        return f"{text}" if len(text) > 10 else f"\n\n\n"  # Space out small figures
    
    elif block_type == "Picture":
        return "\n\n[Image Here]\n\n"  # Placeholder for pictures
    
    elif block_type == "Caption":
        return f"**{text}**\n"  # Bold formatting for captions
    
    elif block_type == "Footnote":
        return "\n"  # New line for footnotes
    
    elif block_type == "Formula":
        return f"## {text}\n" if len(text) > 6 else f"{text}\n"
    
    elif block_type == "List-item":
        return f"- {text}\n"  # List item with bullet point
    
    elif block_type in {"Page-footer", "Page-header"}:
        return "\n"  # Space out page headers and footers
    
    # Default case for text blocks
    return f"{text} " if len(text) > 4 else f"{text}\n"


# Function to save a chunk
def save_chunk(chunk, image, output_path, save_image: bool = False):
    """
    Save a chunk of OCR data to a JSON file and optionally save an image with the chunk layout.

    Parameters:
        - chunk (dict): The chunk of OCR data to be saved.
        - image (PIL.Image.Image): The original image containing the chunk.
        - output_path (str): The path to the directory where the chunk and image will be saved.
        - save_image (bool, optional): Whether to save an image with the chunk layout. Defaults to False.
    Returns:
        None
    """
    # Get the chunk ID and page number
    chunk_id = chunk["meta_data"]["chunk_id"]
    page_num = chunk["meta_data"]["page number"]

    # Define the output directory and file path for the chunk
    chunk_output_dir = f"{output_path}/chunks"
    chunk_file_path = f"{chunk_output_dir}/page_{page_num}_chunk_{chunk_id}.json"

    # Save the chunk data to a JSON file
    with open(chunk_file_path, "w", encoding="utf-8") as chunk_file:
        json.dump(chunk, chunk_file, ensure_ascii=False, indent=4)

    # Optionally save an image with the chunk layout
    if save_image:
        chunk_layout_img = image.copy()
        draw = ImageDraw.Draw(chunk_layout_img)
        # Draw a rectangle around the chunk
        draw.rectangle(
            chunk["meta_data"]["chunk_bbox"],
            outline="red",
            width=3,
        )

        # Save the image with the chunk layout
        chunk_layout_img.save(
            f"{output_path}/chunks_layout/chunk_layout_{chunk['meta_data']['chunk_id']}.png"
        )
