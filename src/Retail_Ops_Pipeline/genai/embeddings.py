import os
from typing import Dict, Any, List
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def textualize_store_data(row: pd.Series) -> str:
    """
    Converts a single store CSV row into a natural language descriptive paragraph.
    
    Args:
        row (pd.Series): A row from the store dataframe.
        
    Returns:
        str: Textual description of the store's profile.
    """
    text = f"Store {int(row['Store'])}: This is a type '{row['StoreType']}' store with assortment level '{row['Assortment']}'. "
    
    if pd.notnull(row['CompetitionDistance']):
        text += f"The nearest competitor is located {row['CompetitionDistance']} meters away. "
        if pd.notnull(row['CompetitionOpenSinceYear']):
            text += f"This competition has been active since month {int(row['CompetitionOpenSinceMonth'])}, year {int(row['CompetitionOpenSinceYear'])}. "
    else:
        text += "There is no recorded nearby competition for this store. "

    if row['Promo2'] == 1:
        text += f"The store participates in continuous promotion (Promo2) since year {int(row['Promo2SinceYear'])}, week {int(row['Promo2SinceWeek'])}. "
        text += f"Promotion intervals are: {row['PromoInterval']}. "
    else:
        text += "The store does not participate in the continuous Promo2 program. "
        
    return text

def create_vector_db() -> None:
    """
    Initializes and saves the FAISS vector database by combining store profiles 
    and general business logic rules.
    """
    try:
        all_documents: List[Document] = []

        # 1. Load & Textualize Tabular Data
        store_csv_path = "data/raw/store.csv"
        if os.path.exists(store_csv_path):
            print(f"[INFO] Processing {store_csv_path} for Tabular RAG...")
            df = pd.read_csv(store_csv_path)
            for _, row in df.iterrows():
                content = textualize_store_data(row)
                doc = Document(
                    page_content=content,
                    metadata={"store_id": int(row['Store']), "type": "store_profile"}
                )
                all_documents.append(doc)
            print(f"[INFO] Successfully textualized {len(df)} store profiles.")

        # 2. Load General Business Rules
        rules_path = "docs/store_rules.txt"
        if os.path.exists(rules_path):
            print(f"[INFO] Loading business rules from {rules_path}...")
            loader = TextLoader(rules_path)
            rules_docs = loader.load()
            for d in rules_docs:
                d.metadata = {"type": "business_rule"}
            all_documents.extend(rules_docs)

        # 3. Create Embeddings (Local HuggingFace Model)
        print("[INFO] Generating embeddings via HuggingFace (all-MiniLM-L6-v2)...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 4. Save to FAISS
        print("[INFO] Constructing FAISS vector index...")
        vector_store = FAISS.from_documents(all_documents, embeddings)
        
        output_dir = "models/vector_store/faiss_index"
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        vector_store.save_local(output_dir)
        
        print(f"[SUCCESS] Tabular RAG Index established at {output_dir}")

    except Exception as e:
        print(f"[ERROR] Knowledge Base Initialization Failed: {str(e)}")

if __name__ == "__main__":
    create_vector_db()
