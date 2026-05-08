<div align="center">
  <h1>🚀 Retail Intelligence MLOps & RAG Pipeline</h1>
  <h3>Enterprise-Grade Sales Forecasting with Tabular RAG and Real-time Observability</h3>

  <p>
    <a href="https://github.com/shivam-nayak-ds"><img src="https://img.shields.io/badge/Maintained%20by-Shivam%20Nayak-blue?style=for-the-badge&logo=github" alt="Maintainer"></a>
    <a href="#"><img src="https://img.shields.io/badge/AWS-EC2%20Deployed-orange?style=for-the-badge&logo=amazon-aws" alt="AWS"></a>
    <a href="#"><img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker" alt="Docker"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python"></a>
  </p>
</div>

---

## 📌 Executive Summary
This project is an end-to-end **Production MLOps Pipeline** designed for retail operations. It transcends traditional static forecasting by integrating **Retrieval-Augmented Generation (RAG)** on Tabular Data. Not only does the system predict future sales using PyTorch/Scikit-Learn models, but it also provides *contextual AI-driven business insights* using Google's Gemini LLM and FAISS vector databases.

The entire infrastructure is fully containerized, deployed on **AWS EC2**, and continuously monitored via **Prometheus and Grafana** for data drift and system health.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Frontend
        S[Streamlit UI]
    end

    subgraph AWS Cloud [AWS EC2 Instance / Docker Compose]
        F[FastAPI Server]
        R[Tabular RAG / FAISS]
        M[ML Predictive Model]
        
        P[Prometheus]
        G[Grafana Dashboard]
        E[Evidently AI - Drift Detection]
    end

    subgraph Data & Automation
        DB[(SQL Database)]
        GH[GitHub Actions CI/CD]
    end

    subgraph External APIs
        LLM[Gemini 1.5 Flash LLM]
    end

    DB -- "Automated Fetch" --> M
    GH -- "Automated Deployment" --> AWS Cloud
    S -- "HTTP POST Request" --> F
    F -- "Fetch Embeddings" --> R
    F -- "Generate Forecast" --> M
    R -- "Context & Queries" --> LLM
    LLM -- "Business Insights" --> F
    
    F -- "Metrics / Logs" --> P
    E -- "Drift Reports" --> P
    P -- "Visualize Health" --> G
```

---

## ✨ Key Enterprise Features

1. **Intelligent Tabular RAG System:** Converts raw CSV rows into high-dimensional semantic stories. Uses `FAISS` to retrieve historical context and `Gemini-1.5-Flash` to explain *why* sales dropped or spiked, giving non-technical managers actionable insights.
2. **End-to-End CI/CD Pipeline:** Fully automated testing and deployment pipelines using **GitHub Actions**, ensuring that production code is always verified before reaching AWS.
3. **Automated Drift Detection & Retraining:** The pipeline continuously monitors for Data and Concept Drift using **Evidently AI**. If the model degrades, it automatically triggers a retraining workflow.
4. **Enterprise SQL Data Ingestion:** Securely fetches raw retail data directly from distributed **SQL Databases** rather than relying on static CSV files.
5. **Real-Time Observability Stack:** Fully integrated **Prometheus** metrics and customized **Grafana** dashboards to monitor API latency, forecast distributions, and system health in real-time.
6. **Cloud-Native Deployment:** Automated infrastructure provisioning via **Terraform** and fully containerized application lifecycle using **Docker Compose** on AWS EC2.

---

## 💻 Technology Stack

| Category | Technologies Used |
|----------|------------------|
| **Core ML** | PyTorch, Scikit-Learn, Pandas, NumPy |
| **Generative AI** | Google Gemini API, LangChain, FAISS |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Observability**| Prometheus, Grafana, Evidently AI, MLflow |
| **Deployment** | Docker, Docker Compose, AWS EC2, Terraform |
| **Automation** | GitHub Actions, SQL |
| **Frontend** | Streamlit |

---

## 🚀 Live Cloud Deployment (AWS)

The core architecture is currently live on an AWS EC2 instance. 
* **FastAPI Swagger UI:** `http://13.200.254.145:8000/docs`
* **Grafana Dashboard:** `http://13.200.254.145:3000` (Login: admin/admin)

*(Note: Live IPs may change. Refer to Terraform outputs if provisioning a new instance).*

---

## 🛠️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/shivam-nayak-ds/Retail-Ops-End-to-End-Sales-Forecasting-Pipeline.git
cd Retail-Ops-End-to-End-Sales-Forecasting-Pipeline
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY="your_gemini_api_key_here"
API_HOST="0.0.0.0"
API_PORT=8000
```

### 3. Launch via Docker Compose (Recommended)
This will spin up the API, Prometheus, and Grafana in detached mode.
```bash
docker-compose up -d --build
```

### 4. Launch Streamlit UI
In a separate terminal, launch the interactive frontend:
```bash
streamlit run app_streamlit.py
```

---
*Built with ❤️ by Shivam Nayak. Bridging the gap between Machine Learning and Business Value.*
