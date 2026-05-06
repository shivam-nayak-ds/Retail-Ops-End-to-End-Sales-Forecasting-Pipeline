import os
from typing import Dict, Any
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Retail_Ops_Pipeline.genai.prompt_templates import RAG_ANALYSIS_PROMPT
from Retail_Ops_Pipeline.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger("RAG_Pipeline")


class RetailRAGPipeline:
    """
    Production-grade RAG pipeline using stable langchain_core chains.
    Architecture: FAISS Retrieval -> Cross-Encoder Reranking -> Groq LLM Analysis
    No fragile AgentExecutor — direct chain invocation for reliability.
    """

    def __init__(self):
        try:
            logger.info("Initializing Retail RAG Pipeline components.")

            # 1. Embedding model (sentence-transformers directly — no wrapper issues)
            self._st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            # 2. FAISS Index via langchain community (stable wrapper)
            index_path = "models/vector_store/faiss_index"
            if not os.path.exists(index_path):
                raise FileNotFoundError(
                    f"FAISS index not found at '{index_path}'. Run embeddings.py first."
                )

            # Use a simple embedding function compatible with FAISS loader
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            self.vector_store = FAISS.load_local(
                index_path,
                _embeddings,
                allow_dangerous_deserialization=True
            )

            # 3. Cross-Encoder for re-ranking (Elite step)
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            # 4. Groq LLM
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3
            )

            # 5. Stable langchain_core chain
            self.prompt = ChatPromptTemplate.from_template(RAG_ANALYSIS_PROMPT)
            self.chain = self.prompt | self.llm | StrOutputParser()

            logger.info("RAG Pipeline (FAISS + CrossEncoder + Groq Chain) loaded successfully.")

        except Exception as e:
            logger.error(f"Initialization Failed: {e}")
            raise

    def get_context(self, query: str, k_retrieve: int = 10, k_return: int = 3) -> str:
        """
        FAISS retrieval + Cross-Encoder re-ranking.
        Returns top-k most relevant store profile chunks.
        """
        try:
            # Step 1: Broad retrieval
            docs = self.vector_store.similarity_search(query, k=k_retrieve)

            # Step 2: Cross-Encoder re-ranking
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)

            # Step 3: Sort by score, return top-k
            ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            top_docs = [doc.page_content for _, doc in ranked[:k_return]]

            return "\n\n".join(top_docs)

        except Exception as e:
            logger.error(f"Context Retrieval Error: {e}")
            return "No store context available."

    def explain_forecast(self, input_data: Dict[str, Any], prediction: float) -> str:
        """
        Main inference method.
        Retrieves store context, then generates business analysis via Groq LLM chain.
        """
        try:
            store_id = input_data.get("Store", "Unknown")
            date = input_data.get("Date", "Unknown Date")

            # Build retrieval query
            query = f"Store {store_id} sales profile assortment competition promo"
            context = self.get_context(query)

            # Invoke stable chain
            analysis = self.chain.invoke({
                "prediction": f"${prediction:,.2f}",
                "input_data": str(input_data),
                "context": context
            })

            return analysis

        except Exception as e:
            logger.error(f"Analysis Generation Failed: {e}")
            return f"Predicted Sales: ${prediction:,.2f}. (AI Analysis unavailable: {e})"


if __name__ == "__main__":
    try:
        pipeline = RetailRAGPipeline()

        test_data = {"Store": 1, "Date": "2026-05-07", "Promo": 1, "DayOfWeek": 4}
        test_prediction = 15420.50

        print("\n" + "=" * 60)
        print("RETAIL RAG PIPELINE — TEST RUN")
        print("=" * 60 + "\n")

        analysis = pipeline.explain_forecast(test_data, test_prediction)
        print(analysis)

    except Exception as e:
        print(f"Test Failed: {e}")
