import os
import threading
import queue


class config:
    RAG_SYSTEM_PROMPT = """You are a conversational AI assistant that is provided a list of documents and a user query to answer based on information from the documents.
    Here is the user query:
    {query}
    """

    SUMM_SYSTEM_PROMPT = """Based on the provided documents, generate a summary for the paragraph that has the provided title.

    Title:
    {title}
    """

    ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    DEVICE = "cuda"

    CHUNKS_QUEUE = queue.Queue()
    EXIST_FLAG = False
    EXIST_FLAG_DOC_ID = None
    OCR_FLAG = False
    EMBED_START_FLAG = threading.Event()

    TOP_K = 3

    DIM = None

    faiss_index = None
    docs_mapping = None
