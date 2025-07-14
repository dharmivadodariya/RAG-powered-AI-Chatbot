from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="RAG DuckDuckGo Chatbot", layout="wide")
st.markdown("<h1 style='text-align: center;'>🔍 RAG DuckDuckGo Chatbot</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
    .block-container {
        max-width: 800px;
        margin: auto;
    }
</style>
""", unsafe_allow_html=True)

st.write("This chatbot uses Retrieval-Augmented Generation (RAG) with DuckDuckGo search and Azure OpenAI to answer your questions. "
         "You can also provide a knowledge base text file to enhance the responses.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 1. Load your knowledge base text
try:
    with open('ipl2025.txt', 'r', encoding='utf-8') as f:
        ipl2025_text = f.read()
except FileNotFoundError:
    ipl2025_text = ""
    st.warning("Knowledge base file 'ipl2025.txt' not found. You can create one to enhance the chatbot's responses.")

# 2. Split into documents if available
documents = []
if ipl2025_text.strip() != "":
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(ipl2025_text)
    documents = [Document(page_content=doc) for doc in docs]

# 3. Embedding model
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
    deployment=os.getenv("EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME"),
    chunk_size=10,
)

# 4. Create FAISS vector database if documents exist
if documents:
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
else:
    retriever = None

# 5. DuckDuckGo search tool
search = DuckDuckGoSearchRun()

# 6. Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=800
)

# 7. RetrievalQA chain (only if retriever exists)
if retriever:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

# Horizontal separator after description
st.markdown("---")

# Display chat history
for i, chat in enumerate(st.session_state.chat_history):
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**Bot:** {chat['content']}")

# Initialize response as None before form
response = None

# 9. Chat input form to avoid session state errors
with st.form("chat_form", clear_on_submit=False):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_query = st.text_input("Ask a question:",  placeholder="Type your question here...")  
    with col2:
        submit = st.form_submit_button("Send")

    if submit and user_query.strip() != "":
        # Append user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # First try retrieval if retriever exists
        response = None
        if retriever:
            with st.spinner("🔍 Retrieving information..."):
                result = qa_chain.invoke({"query": user_query})
            response_text = result["result"]
            
            if response_text and len(response_text.strip()) > 10:
                response = response_text

        # If no good retrieval result, fallback to DuckDuckGoSearch
        if not response:
            with st.spinner("🔎 Searching online..."):
                web_result = search.run(user_query)
            if web_result and len(web_result.strip()) > 0:
                    # Use LLM to summarise web search result
                    prompt = f"""
                    Here is the search result:

                    {web_result}

                    Using this information, provide a **comprehensive, detailed, and helpful answer** to the user's question below. 
                    Include extra background, context, examples, and relevant facts if available, so the user gains a full understanding.

                    User Question: '{user_query}'
                    """

                    llm_response = llm.invoke(prompt)
                    response = llm_response.content
            else:
                response = "❌ Sorry, I couldn't find any information."
            
if response:
    # Append bot response to chat history
    st.session_state.chat_history.append({"role": "bot", "content": response})