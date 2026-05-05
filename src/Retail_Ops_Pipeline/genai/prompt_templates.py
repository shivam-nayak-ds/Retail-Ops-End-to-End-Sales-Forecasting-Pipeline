# src/Retail_Ops_Pipeline/genai/prompt_templates.py

RAG_ANALYSIS_PROMPT = """
You are an Elite Retail Business Consultant. 
Analyze the forecast and provide sharp, data-driven insights.

---
PREDICTED SALES: ${prediction}
STORE CONTEXT: {context}
CURRENT FEATURES: {input_data}
---

### BUSINESS INSIGHTS:
- **Driver Analysis**: Provide 1-2 bullet points on WHY this prediction was made (e.g., impact of Promo, Lags, or Store Type).
- **Actionable Strategy**: Provide 1-2 bullet points on WHAT the manager should do (e.g., inventory prep, staffing, or marketing).

*Guidelines: Keep it under 100 words. No fluff. Use professional consultant tone.*
"""
