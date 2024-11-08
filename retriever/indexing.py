import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from helpers.config import config
from helpers.logger import create_logger
import json

logger = create_logger(__name__, os.getenv("LOGGING_LEVEL", "INFO"))


class IndexDocument:
    def load_model():
        model = SentenceTransformer("BAAI/bge-m3", device=config.DEVICE)  
        config.DIM = model.get_sentence_embedding_dimension()
        logger.info("Embedding model is loaded")
        print("Embedding model is loaded")
        return model

    def create_vector_db(embeddings: list, output_path: str):
        faiss_index = faiss.IndexFlatIP(config.DIM)
        faiss_index.add(np.array(embeddings))

        with open(output_path, "wb") as handle:
            pickle.dump(faiss_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("vector database has been created")
        print("vector database has been created")

        return faiss_index

    def load_vector_db(db_path):
        with open(db_path, "rb") as handle:
            loaded_faiss_index = pickle.load(handle)

        logger.info("vector database has been loaded")
        print("vector database has been loaded")

        return loaded_faiss_index

    # def index(model):
    #     config.EMBED_START_FLAG.wait()
    #     print("Embedding Start Flag:", config.EXIST_FLAG)
    #     if config.EXIST_FLAG:
    #         db_path = os.path.join(
    #             "output", config.EXIST_FLAG_DOC_ID, "vector_db", "db.pickle"
    #         )
    #         config.faiss_index = IndexDocument.load_vector_db(db_path)

    #         # load chunks
    #         chunks_parent_path = os.path.join(
    #             "output", config.EXIST_FLAG_DOC_ID, "chunks"
    #         )
    #         print("Parent Chunks_path:", chunks_parent_path)
    #         chunks_paths = os.listdir(chunks_parent_path)
    #         docs_mapping = []
    #         for chunk in chunks_paths:
    #             with open(os.path.join(chunks_parent_path, chunk), "r") as f:
    #                 data = json.load(f)
    #                 docs_mapping.append(data)
    #         config.docs_mapping = docs_mapping

    #     else:
    #         docs_embeddings = []
    #         docs_mapping = []
    #         print("Start indexing")
    #         print("Chunks Queue:", config.CHUNKS_QUEUE.empty())
    #         print("OCR Flag:", config.OCR_FLAG)
    #         while not (config.CHUNKS_QUEUE.empty() and config.OCR_FLAG):
    #             doc = config.CHUNKS_QUEUE.get()
    #             embedding = model.encode([doc], convert_to_tensor=True)[0]
    #             docs_embeddings.append(embedding.cpu().numpy())
    #             docs_mapping.append(doc)
    #             print("Doc:", doc)
    #             print("Embedding:", embedding)
    #             if config.CHUNKS_QUEUE.empty() and config.OCR_FLAG:
    #                 config.CHUNKS_QUEUE.task_done()
    #         else:
    #             output_path = os.path.join(
    #                 "output", config.EXIST_FLAG_DOC_ID, "vector_db", "db.pickle"
    #             )

    #         print("Creating vector database")
    #         config.faiss_index = IndexDocument.create_vector_db(
    #             docs_embeddings, output_path
    #         )
    #         print("Vector database created")
    #         config.docs_mapping = docs_mapping


    def index(model):
        config.EMBED_START_FLAG.wait()
        print("Embedding Start Flag:", config.EXIST_FLAG)
    
        if config.EXIST_FLAG:
            db_path = os.path.join(
                "output", config.EXIST_FLAG_DOC_ID, "vector_db", "db.pickle"
            )
            config.faiss_index = IndexDocument.load_vector_db(db_path)

            # load chunks
            chunks_parent_path = os.path.join(
                "output", config.EXIST_FLAG_DOC_ID, "chunks"
            )
            print("Parent Chunks_path:", chunks_parent_path)
            chunks_paths = os.listdir(chunks_parent_path)
            docs_mapping = []
        
            for chunk in chunks_paths:
                # Skip the .ipynb_checkpoints directory or any file inside it
                if chunk == ".ipynb_checkpoints" or os.path.isdir(os.path.join(chunks_parent_path, chunk)):
                    continue

                with open(os.path.join(chunks_parent_path, chunk), "r") as f:
                    data = json.load(f)
                    docs_mapping.append(data)
        
            config.docs_mapping = docs_mapping

        else:
            docs_embeddings = []
            docs_mapping = []
            print("Start indexing")
            print("Chunks Queue:", config.CHUNKS_QUEUE.empty())
            print("OCR Flag:", config.OCR_FLAG)
        
            while not (config.CHUNKS_QUEUE.empty() and config.OCR_FLAG):
                doc = config.CHUNKS_QUEUE.get()
                embedding = model.encode([doc], convert_to_tensor=True)[0]
                docs_embeddings.append(embedding.cpu().numpy())
                docs_mapping.append(doc)
                print("Doc:", doc)
                print("Embedding:", embedding)
                if config.CHUNKS_QUEUE.empty() and config.OCR_FLAG:
                    config.CHUNKS_QUEUE.task_done()
            else:
                output_path = os.path.join(
                    "output", config.EXIST_FLAG_DOC_ID, "vector_db", "db.pickle"
                )

            print("Creating vector database")
            config.faiss_index = IndexDocument.create_vector_db(
                docs_embeddings, output_path
            )
            print("Vector database created")
            config.docs_mapping = docs_mapping


    def search(query, model, faiss_index, docs_mapping):
        print("Docs Mapping:", docs_mapping)
        print("query:", query)
        query_embed = model.encode([query], convert_to_tensor=True)
        print("query_embed:", query_embed)
        scores, ids = faiss_index.search(query_embed.cpu().numpy(), config.TOP_K)
        print("Scores:", scores)
        print("IDs:", ids)

        return [
            {
                "page_content": docs_mapping[id]["page_content"],
                "meta_data": docs_mapping[id]["meta_data"],
                "confidence": score,
            }
            for score, id in zip(scores[0], ids[0])
        ]
