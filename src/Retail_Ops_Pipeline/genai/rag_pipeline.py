import os
from typing import Dict, Any, List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from Retail_Ops_Pipeline.genai.prompt_templates import RAG_ANALYSIS_PROMPT
from Retail_Ops_Pipeline.utils.logger import get_logger
from dotenv import load_dotenv

# Environment variables load
load_dotenv()
logger = get_logger("RAG_Pipeline")

class RetailRAGPipeline:
    def __init__(self):
        """
        Step 1: Initialize pipeline components and load models.
        """
        try:
            logger.info("Initializing Retail RAG Pipeline components.")
            
            # 1. Load Embedding Model (Must match the one used in indexing)
            # Using HuggingFace locally for stability as per previous successful tests
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # 2. Load FAISS Index
            index_path = "models/vector_store/faiss_index"
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run embeddings.py first.")
            
            self.vector_store = FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # 3. Load Cross-Encoder for Re-ranking (Elite Step)
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # 4. Initialize Groq LLM for analysis
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3
            )
            
            # 5. Setup Prompt Template
            from Retail_Ops_Pipeline.genai.prompt_templates import RAG_ANALYSIS_PROMPT
            self.prompt = PromptTemplate(
                template=RAG_ANALYSIS_PROMPT,
                input_variables=["prediction", "input_data", "context"]
            )
            
            logger.info("RAG Pipeline components (with Re-ranker) loaded successfully.")

        except Exception as e:
            logger.error(f"Initialization Failed: {str(e)}")
            raise

    def get_context(self, store_id: int, query: str) -> str:
        """
        Retrieves store-specific profile and re-ranks them for maximum relevance.
        """
        try:
            # 1. Initial Retrieval (Get more docs for re-ranking)
            initial_docs = self.vector_store.similarity_search(query, k=10)
            
            # 2. Prepare pairs for Re-ranking [query, doc_text]
            pairs = [[query, doc.page_content] for doc in initial_docs]
            
            # 3. Get relevance scores from Cross-Encoder
            scores = self.reranker.predict(pairs)
            
            # 4. Sort documents by scores in descending order
            reranked_docs = [doc for _, doc in sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)]
            
            # 5. Take top 3 most relevant docs
            context = "\n".join([doc.page_content for doc in reranked_docs[:3]])
            return context
        except Exception as e:
            logger.error(f"Context Retrieval/Re-ranking Error: {str(e)}")
            return "No specific context found."

    def explain_forecast(self, input_data: Dict[str, Any], prediction: float) -> str:
        """
        Orchestrates retrieval, re-ranking, and generation.
        """
        try:
            store_id = input_data.get("Store")
            if not store_id:
                return "Error: Store ID missing in input data."

            # 1. Retrieve & Re-rank Knowledge
            # We use a query that describes what we are looking for
            query = f"Business rules and profile for Store {store_id}"
            context = self.get_context(store_id, query)
            
            # 2. Prepare RAG Chain
            chain = self.prompt | self.llm
            
            # 3. Generate Analysis
            response = chain.invoke({
                "prediction": f"{prediction:,.2f}",
                "input_data": str(input_data),
                "context": context
            })
            
            return response.content

        except Exception as e:
            logger.error(f"Analysis Generation Failed: {str(e)}")
            return f"Predicted Sales: {prediction:,.2f}. (AI Analysis unavailable)."

if __name__ == "__main__":
    # Test Block for Manual Verification
    try:
        pipeline = RetailRAGPipeline()
        # Mock Input
        test_data = {"Store": 1, "Promo": 1, "DayOfWeek": 1}
        test_prediction = 15420.50
        
        print("\n" + "="*50)
        print("RUNNING RAG ANALYSIS TEST")
        print("="*50 + "\n")
        
        analysis = pipeline.explain_forecast(test_data, test_prediction)
        print(analysis)
        
    except Exception as e:
        print(f"Test Failed: {e}")
