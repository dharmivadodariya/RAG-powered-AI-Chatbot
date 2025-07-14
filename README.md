"# RAG-powered-AI-Chatbot" 

A **Streamlit-based chatbot** that uses **Retrieval-Augmented Generation (RAG)** with a local knowledge base (`ipl2025.txt`), **DuckDuckGo Web Search**, and **Azure OpenAI** to provide accurate and detailed answers to user queries.

---

## Features

- Chat interface powered by **Streamlit**
- Supports user-provided knowledge base via `ipl2025.txt`
- Integrates **FAISS** vector database for fast similarity search
- Falls back to **DuckDuckGo Search** when local knowledge is insufficient
- Uses **Azure OpenAI GPT** for natural language generation
- Smartly decides when to use retrieval vs. online search

---

## Requirements

- Python 3.8+
- GitHub repository with the following structure:

📁 Chatbot/
│
├── chatbot.py
├── .env
└── ipl2025.txt (optional)

---

## Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. **Install dependencies**

    pip install -r requirements.txt

3. **Set up your .env file**

    Create a .env file with the following variables:

    AZURE_OPENAI_ENDPOINT=your_endpoint
    AZURE_OPENAI_API_VERSION=2023-05-15
    AZURE_OPENAI_API_KEY=your_key
    AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
    EMBEDDING_AZURE_OPENAI_ENDPOINT=your_embedding_endpoint
    EMBEDDING_AZURE_OPENAI_API_VERSION=2023-05-15
    EMBEDDING_AZURE_OPENAI_API_KEY=your_embedding_key
    EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME=your_embedding_deployment

## Run the App

streamlit run chatbot.py

## Example Use Cases

AI-powered Q&A over custom documents

Supplemented chat responses with up-to-date web search

Internal tools for domain-specific knowledge + general search