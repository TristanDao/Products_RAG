from embedding import EmbeddingModel
from vector_db import VectorDatabase
from semetic_router.samples import chitchatSample, productsSample
from semetic_router.route import Route, SemanticRouter
from reflection import Reflection

from openai import OpenAI
from google import genai

from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
load_dotenv()

class RAGSystem:
    def __init__ (
            self,
            db_type : str = "mongodb",
            embedding_provider: str = "huggingface",
            llm_provider: str = "gemini",
            embedding_model: Optional[str] = None,
            llm_model: Optional[str] = None,
            collection_name: str = "products",
            routes: Optional[List[Route]] =None
    ):
        self.collection_name = collection_name

        #Inititalize Vector Database
        print(f" Initializing {db_type} database...")
        self.vector_db = VectorDatabase(db_type= db_type)
        print(f"connected success!")


        #Innitalize Embedding Model
        print(f"Initializing {embedding_provider} embedding ...")
        self.embedding_model = EmbeddingModel(provider = embedding_provider, model_name=embedding_model)
        print(f"connected success!")
        
        #Initialze LLM
        print(f"Initializing {llm_provider} LLM...")
        try:
            self.llm_provider = llm_provider.lower()

            if self.llm_provider == "openai":
                self.llm_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
                self.llm_model = llm_model or "gpt-4o-mini"
            elif self.llm_provider =="gemini":
                self.llm_client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
                self.llm_model = llm_model or "gemini-2.0-flash-exp"
            else: 
                raise ValueError (f"Unsupported LLM provider: {self.llm_provider}")
                return
            print("LLM success!\n")
        except Exception as e:
            print(f"Init failed! {e}")
            return
        print("Khởi tạo Semantic Router và Reflection...")
        self.reflection = Reflection(self.llm_client)
        if routes:
            self.router = SemanticRouter(self.embedding_model, routes)
        else:
            self.router = None
        print("Hệ thống RAG đã sẵn sàng!")
    #Hàm phân loại trò chuyện hay hỏi sản phẩm
    def route_query(self, query: str, message: List[Dict]) -> tuple[str,str]:
        if not self.router:
            return "products"
        rewritten_query =self.reflection.rewrite(message, query)
        print(f"Câu hỏi độc lập: {rewritten_query}")

        route_result = self.router.guide(rewritten_query)
        best_route = route_result[1]
        print(f"Phân loại Semantic Route: {best_route}")
        return best_route, rewritten_query

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str,Any]]:
        query_embedding = self.embedding_model.encode_single(query)
        results = self.vector_db.query(
            collection_name= self.collection_name,
            embedding_vector=query_embedding,
            top_k = top_k
        )
        return results
    
    def format_context(self, results: List[Dict[str,Any]]) -> str:
        context_parts = []
        for i, result in enumerate(results, 1):
            doc = result['information']
            context_parts.append(f"Tài liệu {i}:\n{doc}\n")
        return "\n".join(context_parts)

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"""Bạn là một nhân viên tư vấn online chuyên nghiệp và nhiệt tình, cho của hàng hàng bán điện thoại của Shop "Thịnh Đào.
Hãy trả lời câu hỏi của khách hàng dựa trên thông tin được cung cấp.

Thông tin sản phẩm:
{context}

Câu hỏi của khách hàng: {query}

Hướng dẫn trả lời:
- Trả lời chính xác, chi tiết dựa trên thông tin được cung cấp
- Nếu có nhiều sản phẩm phù hợp, hãy so sánh và gợi ý
- Nêu rõ giá cả, ưu đãi nếu có
- Thân thiện và chuyên nghiệp
- Nếu không có thông tin, hãy thừa nhận thay vì bịa đặt

Trả lời: """
        if self.llm_provider == "openai":
            response = self.llm_client.chat.completions.create(
                model = self.llm_model,
                messages = [
                    {"role": "system", "content": "Bạn là một nhân viên tư vấn bán hàng online chuyên nghiệp về điện thoại."},
                    {"role": "user", "content": prompt}
                ],
                temperature= 0.7,
                max_tokens= 1000
            )
            return response.choices[0].message.content
        
        elif self.llm_provider == "gemini":
            response = self.llm_client.models.generate_content(
                model = self.llm_model,
                contents= prompt
            ) 
            return response.text
        else:
            raise ValueError(f"LLM provider '{self.llm_provider}' không được hỗ trợ.")

    def query(self, user_query: str, top_k: int = 5, messages: Optional[List[Dict]]= None) -> Dict[str, Any]:
        if not messages:
            messages = [{"role": "system",
                         "content": "Bạn là một nhân viên tư vấn bán hàng online chuyên nghiệp về điện thoại."}]
        #Phân loại truy vấn + viết lại
        best_route, rewritten_query = self.route_query(user_query, messages) 
        if best_route == "chitchat":
            response = self.llm_client.chat.completions.create(
                model= self.llm_model,
                messages= messages + [{"role": "user", "content": rewritten_query}]
            )      
            return {"answer": response.choices[0].message.content}
        
        else: #product
            #Step 1: Retrieve relevant documents
            results = self.retrieve(rewritten_query, top_k=top_k)
            #Step 2: Format context
            context = self.format_context(results)
            #Step3: Generate response
            answer = self.generate_response(rewritten_query, context)
            return {
                "answer": answer,
                "source": results,
                "query": rewritten_query,
                "context": context
            }

def main():
    routes = [Route(name ="products", samples = productsSample),
              Route(name ="chitchat", samples= chitchatSample)]
    rag = RAGSystem(
        db_type= "mongodb",
        embedding_provider= "huggingface",
        llm_provider= "openai",
        collection_name= "products",
        routes= routes
    )       
    #Interactice chat mode
    print("\n"+ "="*60)
    print("RAG Chatbot - Hỏi đáp về sản phẩm")
    print("="*60)
    print("Nhập 'exit' hoặc 'quit' để thoát \n")

    messages =[]
    while True:
        try:
            user_input = input("Bạn: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'thoat', 'thoát']:
                print("\n Tạm biệt! Hẹn gặp lại.")
                break
            result = rag.query(user_query=user_input, messages=messages)
            print(f"\n Trợ lý: {result['answer']} \n")
            print("-"*60)
            messages.append({"role":"user", "content": user_input})
            messages.append({"role":"assistant", "content": result["answer"]})

        except KeyboardInterrupt:
            print("\n\n Tạm biệt!")
            break
        except Exception as e:
            print(f"\n Lỗi: {e}\n")

if __name__ == "__main__":
    main()

