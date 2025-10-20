from openai import OpenAI
from sentence_transformers import SentenceTransformer
from google import genai
import torch
import requests

import os
from typing import List, Union, Optional
from dotenv import load_dotenv
load_dotenv()

class EmbeddingModel:
    def __init__ (self, provider: str, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()

        if self.provider == "openai":
            self.client = OpenAI(api_key = self.api_key)
            self.model_name = model_name or "text-embedding-3-small"

        elif self.provider == "gemini":
            self.client = genai.Client(api_key = self.api_key)
            self.model_name = model_name or  "gemini-embedding-001"

        elif self.provider == "huggingface":
            self.model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            #Đường dẫn model cục bộ trong dự án
            local_model_path = os.path.join("models",os.path.basename(self.model_name))
            # Nếu chưa có thì tải và lưu lại
            if not os.path.exists(local_model_path):
                print(f"Tải model từ Hugging Face: {self.model_name}")
                model = SentenceTransformer(self.model_name, device= device)
                os.makedirs("models", exist_ok= True)
                model.save(local_model_path)
                print(f"Model đã lưu tại: {local_model_path}")
            #Sau này chỉ load local
            print(f"Model đang được load từ local: {local_model_path}")
            self.model = SentenceTransformer(self.model_name, device = device)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_api_key(self) -> Optional[str]:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider =="gemini":
            return os.getenv("GEMINI_API_KEY")
        return None
    def get_vector_size(self) -> int:
        if self.provider == 'openai':
            if "text-embedding-3-small" in self.model_name:
                return 1536
            elif 'text-embedding-3-large' in self.model_name:
                return 3072
            return 1536
        elif self.provider == "gemini":
            if "gemini-embedding-001" in self.model_name:
                return 3072
            return 768

        elif self.provider == "huggingface":
            return self.model.get_sentence_embedding_dimension()
        return 1536

    def encode(self, texts: Union[str,List[str]]) -> List[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValueError("Input should be a string or a list of strings.")
    
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input = texts, 
                model = self.model_name 
            )
            embeddings = [data.embedding for data in response.data]
            return embeddings
        elif self.provider == "gemini":
            response = self.client.models.embed_content(
                model = self.model_name,
                contents = texts
            )
            if response.embeddings is None:
                return []
            return [data.values for data in response.embeddings if data and data.values is not None]

        elif self.provider == "huggingface":
            embeddings = self.model.encode(texts, convert_to_tensor = False)
            return embeddings.tolist()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def encode_single(self, text: str) -> List[float]:
        result = self.encode(text)
        return result[0] if result else []
        
def main():
    embedder = EmbeddingModel(provider = "huggingface")
    texts = ["Hello, world!", "How are you?"]
    embeddings = embedder.encode(texts)
    print(f"{embeddings}")
    for i,eb in enumerate(embeddings):
        print(f"Text: {texts[i]} \nEmbedding: {eb[:5]}...\n")

if __name__ == "__main__":
    main()


            





