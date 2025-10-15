from embedding import EmbeddingModel
from vector_db import VectorDatabase

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
            embedding_provider: str = "gemini",
            llm_provider: str = "gemini",
            embedding_model: Optional[str] = None,
            llm_model: Optional[str] = None,
            collection_name: str = "products"
    ):
        self.collection_name = collection_name

        #Inititalize Vector Database
        print(f" Inititalizing {db_type} database...")
        try:
            self.vector_db = VectorDatabase(db_type= db_type)
            print(f"connected success!")
        except Exception as e:
            print(f"Init faile! NOTE: {e}")
            return

        #Innitalize Embedding Model
        print(f"Initializing {embedding_provider} embedding ...")
        try:
            self.embedding_model = EmbeddingModel(provider = embedding_provider, model_name=embedding_model)
            print(f"connected success!")
        except Exception as e:
            print(f"Init faile! NOTE: {e}")
            return
        
        #Initialze LLM
        print(f"Initializing {llm_provider} LLM...")
        try:
            self.llm_provider = llm_provider.lower()

            if self.llm_provider == "openai":
                self.llm_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
                self.llm_model = llm_model or "gpt-04-mini"
            elif self.llm_provider =="gemini":
                self.llm_client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
                self.llm_model = llm_model or "gemini-2.0-flash-exp"
            else: 
                raise ValueError (f"Unsupported LLM provider: {self.llm_provider}")
                return
            print("RAG System initialized successfully!\n")
        except Exception as e:
            print(f"Init failed! {e}")
            return
    
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

    def query(self, user_query: str, top_k: int = 5, verbose: bool= True) -> Dict[str, Any]:
        if verbose: 
            print(f"\n{'='*60}")
            print(f"Query: {user_query}")
            print(f"{'='*60}\n")
            print(f"Retrieving relevant documents...")
        #Step 1: Retrieve relevant documents
        results = self.retrieve(user_query, top_k=top_k)
        if not results:
            return {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn",
                "source": [],
                "query": user_query
            }
        
        if verbose:
            print(f"Found {len(results)} relevant documents\n")
        #Step 2: Format context
        context = self.format_context(results)

        #Step3: Generate response
        if verbose:
            print("Generate response...")
        answer = self.generate_response(user_query, context)

        if verbose:
            print(f"Response generated\n")
            print(f"{'='*60}")
            print(f"Answer: \n{answer}")
            print(f"{'='*60}\n")
        
        return {
            "answer": answer,
            "source": results,
            "query": user_query,
            "context": context
        }
    
    def chat(self):
        print("\n"+ "="*60)
        print("RAG Chatbot - Hỏi đáp về sản phẩm")
        print("="*60)
        print("Nhập 'exit' hoặc 'quit' để thoát \n")

        while True:
            try:
                user_input = input("Bạn: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['exit', 'quit', 'thoat', 'thoát']:
                    print("\n Tạm biệt! Hẹn gặp lại.")
                    break
                result = self.query(user_input, verbose= False)
                print(f"\n Trợ lý: {result['answer']} \n")
                print("-"*60)

            except KeyboardInterrupt:
                print("\n\n Tạm biệt!")
                break
            except Exception as e:
                print(f"\n Lỗi: {e}\n")
def main():
    rag = RAGSystem(
        db_type= "mongodb",
        embedding_provider= "gemini",
        llm_provider= "gemini",
        collection_name= "products"
    )       

    print("\n" + "="*60)
    print("Emxample Query")
    print("="*60)

    result = rag.query(
        "Tôi muốn mua điện thoại IPhone giá khoảng 20 triệu, có những lưa chọn nào?",
        top_k =5
    )

    print("\n Sources used:")
    for i, source in enumerate(result['source'][:3],1):
        title = source.get('title') or source.get('metadata',{}).get('title',"Unknow")
        score = source.get('score', source.get('distance', 'N/A'))
        print(f" {i}.{title} (score: {score})")
    
    #Interactice chat mode
    print("\n" + "="*60)
    print("Starting interactive chat mode...")
    print("="*60)
    rag.chat()

if __name__ == "__main__":
    main()

