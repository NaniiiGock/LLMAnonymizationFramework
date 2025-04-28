from src.database.chroma_db import ChromaDBManager
from src.database.mongo_db import MongoHEVectorStore


class DBManager:
    def __init__(self, config={}):
        self.config = config
        self.db_provider = self.config.get("vector_store")
        self.setup_db()
    def setup_db(self):
        if self.db_provider == "chroma":
            self.db = ChromaDBManager()
        else:
            self.db = MongoHEVectorStore()
        
    def run_retriever(self, query, folder="uploaded_files"):
        result = self.db.run_retriever(query, folder)
        return result

