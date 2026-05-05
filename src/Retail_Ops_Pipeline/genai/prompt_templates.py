# src/Retail_Ops_Pipeline/genai/prompt_templates.py

RAG_ANALYSIS_PROMPT = """
You are an expert Retail Operations Analyst. 
Based on the following data and business rules, provide a professional analysis for the Store Manager.

---
PREDICTED SALES: {prediction}
INPUT DATA: {input_data}
CONTEXT FROM KNOWLEDGE BASE: {context}
---

Your analysis must include:
1. A brief summary of the forecast.
2. The key drivers behind this number (based on data and context).
3. At least two actionable business insights for the Store Manager.

Format: Return a clean, professional response in plain text.
"""
