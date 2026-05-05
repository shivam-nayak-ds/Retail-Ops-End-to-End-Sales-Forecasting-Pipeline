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
            
            # 3. Initialize Groq LLM for analysis
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3
            )
            
            # 4. Setup Prompt Template
            self.prompt = PromptTemplate(
                template=RAG_ANALYSIS_PROMPT,
                input_variables=["prediction", "input_data", "context"]
            )
            
            logger.info("RAG Pipeline components loaded successfully.")

        except Exception as e:
            logger.error(f"Initialization Failed: {str(e)}")
            raise

    def get_context(self, store_id: int) -> str:
        """
        Retrieves store-specific profile and relevant business rules from FAISS.
        """
        try:
            # Search for specific store profile
            # Our indexing format: "Store {id}: ..."
            query = f"Store {store_id}"
            docs = self.vector_store.similarity_search(query, k=3)
            
            context = "\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            logger.error(f"Context Retrieval Error: {str(e)}")
            return "No specific context found."

    def explain_forecast(self, input_data: Dict[str, Any], prediction: float) -> str:
        """
        Orchestrates retrieval and generation to provide a business explanation.
        """
        try:
            store_id = input_data.get("Store")
            if not store_id:
                return "Error: Store ID missing in input data."

            # 1. Retrieve Contextual Knowledge
            context = self.get_context(store_id)
            
            # 2. Prepare RAG Chain
            chain = self.prompt | self.llm
            
            # 3. Generate Analysis via Gemini
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
