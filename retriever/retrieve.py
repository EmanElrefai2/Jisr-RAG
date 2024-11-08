import numpy as np
from sentence_transformers import SentenceTransformer, util
import pickle
import os
import torch
from collections import deque
from helpers.s3_downloader import download_s3

class DocumentRetriever:
    def __init__(self, db_path='vector_db.pkl', max_queue_size=1000):
        self.model = None
        self.db_path = db_path
        self.db = None
        self.passage_embeddings = deque(maxlen=max_queue_size)
        self.load_or_create_db()

    def initialize_model(self, bucket_name, model_key, local_model_path='./custom_model'):
        """
        Initialize the model by downloading it from S3 and loading it.
        
        :param bucket_name: Name of the S3 bucket
        :param model_key: S3 key of the model file
        :param local_model_path: Local path to save and load the model
        """
        local_file_path = download_s3(bucket_name, model_key, local_model_path)
        if local_file_path:
            try:
                self.model = SentenceTransformer(local_model_path)
                print("Custom model loaded successfully.")
            except Exception as e:
                print(f"Error loading the model: {e}")
        else:
            print("Failed to initialize the model due to download error.")

    def load_or_create_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.db = pickle.load(f)
            for doc in self.db:
                self.passage_embeddings.append(torch.tensor(doc['embedding']))
        else:
            self.db = []

    def index_documents(self, documents):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        for doc in documents:
            content = doc['page_content']
            metadata = doc['meta_data']
            embedding = self.model.encode(content, convert_to_tensor=True)
            self.passage_embeddings.append(embedding)
            self.db.append({
                'content': content,
                'metadata': metadata,
                'embedding': embedding.cpu().numpy()
            })
        
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.db, f)

    def semantic_search(self, query, top_k=3):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        question_embedding = self.model.encode(query, convert_to_tensor=True)
        passage_embeddings_tensor = torch.stack(list(self.passage_embeddings))
        hits = util.semantic_search(question_embedding, passage_embeddings_tensor, top_k=top_k)
        hits = hits[0]
        
        result =  [
                {
                    "confidence": hit['score'],
                    "page_content": self.db[hit['corpus_id']]['content'],
                    "metadata": self.db[hit['corpus_id']]['metadata']
                } 
                for hit in hits
            ]
    
        return result