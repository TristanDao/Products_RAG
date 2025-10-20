from typing import List, Optional
from embedding import EmbeddingModel
import numpy as np

class Route:
    def __init__(self,name: Optional[str]= None,samples: Optional[str]= None ):
        self.name = name
        self.samples = samples or []

class SemanticRouter():
    def __init__(self, embedding: EmbeddingModel, routes: List[Route]):
        self.routes = routes
        self.embedding = embedding
        self.routesEmbedding = {}

        for route in self.routes:
            all_embs = []
            batch_size = 100  # Giới hạn theo API của Gemini

            # Chia nhỏ danh sách samples thành nhiều batch
            for i in range(0, len(route.samples), batch_size):
                batch = route.samples[i:i + batch_size]
                emb = self.embedding.encode(batch)
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                all_embs.append(emb)

            # Gộp tất cả embeddings lại
            self.routesEmbedding[route.name] = np.vstack(all_embs)
    
    def get_routes(self):
        return self.routes
    
    def guide(self, query: str) :
        queryEmbedding = self.embedding.encode([query])
        queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding)

        scores =[]
        for route in self.routes:
            emb = self.routesEmbedding[route.name]
            sims = np.dot(emb, queryEmbedding.T).flatten()
            score = float(np.mean(sims))
            scores.append((score, route.name))

        scores.sort(key =lambda x:x[0],reverse=True)
        return scores[0]