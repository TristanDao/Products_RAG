from pymongo import MongoClient
import certifi
from chromadb import HttpClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

from supabase import create_client

from dotenv import load_dotenv

import os
load_dotenv()

# from langchain.vectorstores import MongoDBAtlasVectorSearch
# from langchain.embeddings.base import Embeddings

from typing import List, Dict, Any


class VectorDatabase:
    def __init__(self, db_type: str):
        self.db_type = db_type.lower()
        self.client = None

        if self.db_type == "mongodb":
            mongodb_uri = os.getenv("MONGODB_URI")
            if not mongodb_uri :
                raise ValueError("MongoDB environment variables are not set properly.")
            self.client = MongoClient(mongodb_uri, tls = True, tlsCAFile = certifi.where())

        elif self.db_type == "chromadb":
            chromadb_host = os.getenv("CHROMADB_HOST")
            chromadb_port = os.getenv("CHROMADB_PORT")
            if not chromadb_host  or not chromadb_port:
                raise ValueError("ChromaDB environment variables are not set properly.")
            self.client = HttpClient(host = chromadb_host, port = int(chromadb_port))

        elif self.db_type == "qdrant":
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_key = os.getenv("QDRANT_KEY")
            if not qdrant_url or not qdrant_key:
                raise ValueError("Missing Qdrant enviroment variables.")
            self.client = QdrantClient(url = qdrant_url, api_key = qdrant_key)

        elif self.db_type == "supabase":
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase environment variables are not set properly.")
            self.client  = create_client(supabase_url, supabase_key)
        else:
            raise ValueError("Unsupported database type")
    def create_collection(self, collection_name: str, vector_size: int = 1536):
        if self.db_type == "qdrant":
            try:
                self.client.create_collection(
                    collection_name = collection_name,
                    vectors_config = VectorParams(size = vector_size, distance = Distance.COSINE)
                )
                print(f"Create Qdrant collectionL {collection_name}")
            except Exception as e:
                print(f"Collection already exists or error: {e}")
        elif self.db_type == "chromadb":
            try:
                self.client.get_or_create_collection(name=collection_name)
                print(f'Create/Retrieved chormaDB collection: {collection_name}')
            except:
                print(f"Error creating collection: {e}") 
        elif self.db_type == "mongodb":
            print(f"Mongo collection will be create automatically: {collection_name}")
        elif self.db_type == 'supabase':
            print(f'Using existing Supabase table: {collection_name}')
        else: 
            raise ValueError(f"Unsupport database type:{self.db_type}")

    def document_exists(self,collection_name: str, filter_dict: Dict[str, Any]) -> bool:
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            return collection_name.find_one(filter_dict) is not None
        elif self.db_type == "chromadb":
            try:
                collection = self.client.get_collection(name = collection_name)
                results = collection.get(where = filter_dict)
                return len(results['ids']) > 0
            except Exception:
                return False   
        elif self.db_type == "qdrant":
            try:
                scroll_result = self.client.scroll(
                    collection_name = collection_name,
                    scroll_filter = {"must": [{"key": k, "match": {"value":v}} for k, v in filter_dict.items()]},
                    limit = 1
                )
                return len(scroll_result[0]) > 0
            except Exception:
                return False
        elif self.db_type == "supabase":
            try:
                query = self.client.table(collection_name).select("*")
                for key, value in filter_dict.items():
                    query = query.eq(key, value)
                response = query.limit(1).execute()
                return bool(response.data and len(response.data)>0)
            except Exception:
                return False    
            
    def insert(self, data: List[Dict[str, Any]], collection_name: str):
        if not data:
            raise ValueError("No data provided for insertion")
        
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            collection.insert_many(data)
            
        elif self.db_type == "chromadb":
            collection = self.client.get_or_create_collection(name = collection_name)
            for record in data:
                collection.add(
                    ids = [record["id"]],
                    embeddings = [record["embedding"]],
                    metadatas = [record.get("metadata",{})],
                    documents = [record.get("document", "")]
                )
        elif self.db_type == "qdrant":
            point = [
                PointStruct(
                    id = record["id"],
                    vector = record["embedding"],
                    payload = record.get("metadata", {}),
                ) for record in data
            ]
            self.client.upsert(
                collection_name = collection_name,
                points = point
            )
        elif self.db_type == "supabase":
            self.client.table(collection_name).insert(data).execute()
        else:
            raise ValueError("Unsupported database type")
    def query(self, collection_name: str,embedding_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            pipeline = [
                {
                    "$vectorSearch":{
                        "index": "vector_index",
                        "path": "embedding",
                        "numCandidates" : 100,
                        "queryVector": embedding_vector,
                        "limit": top_k
                    }
                },
                {"$project": {"_id": 0, "score": {"$meta": "vectorSearchScore"}, "document": 1, "metadata": 1,"information": 1}}
            ]
            results = list(collection.aggregate(pipeline))
            return results
        elif self.db_type == "chromadb":
            collection = self.client.get_or_create_collection(name = collection_name)
            results = collection.query(
                query_embeddings = [embedding_vector],
                n_results = top_k
            )
            output = []
            for i in range(len(results['ids'][0])):
                output.append({
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
            return output
        
        elif self.db_type == "qdrant":
            search_result = self.client.search(
                collection_name = collection_name,
                query_vector = embedding_vector,
                limit = top_k
            )
            output = []
            for result in search_result:
                output.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            return output
        elif self.db_type == "supabase":
            response = (
                self.client.rpc(
                    "match_vectors",
                    {"query_vector": embedding_vector, "match_count": top_k}
                ).execute()
            )
            if response.error:
                raise Exception(f"Supabase query error: {response.error.message}")
            return response.data
        else:
            raise ValueError("Unsupported database type")
    def delete_collection(self, collection_name: str):
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            db[collection_name].drop()
        elif self.db_type == "chromadb":
            self.client.delete_collection(name = collection_name)
        elif self.db_type == "qdrant":
            self.client.delete_collection(collection_name = collection_name)
        elif self.db_type == "supabase":
            print("Warning: Supabase colleciton deletion must be done via dashboard")
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            return{
                "count": collection.count_documents({}),
                "name": collection_name
            }
        elif self.db_type == "chromadb":
             collection = self.client.get_collection(name= collection_name)
             return{
                 "count": collection.count(),
                 "name": collection_name
             }
        elif self.db_type == "qdrant":
            info = self.client.get_collection(collection_name = collection_name)
            return{
                "count": info.points_count,
                "name": collection_name,
                "vector_size": info.config.params.vectors.size
            }
        elif self.db_type == "supabase":
            response = self.client.table(collection_name).select("*",count = "exact").limit(0).execute()
            return {
                "count": response.count,
                "name": collection_name
            }
        return {}

# def main():
#     try:
#         vector_db = VectorDatabase(db_type="qdrant")
#         print("Succussed")
#     except Exception as e:
#         print(f"Failed!: {e} ")

# if __name__ == "__main__":
#     main()      
        