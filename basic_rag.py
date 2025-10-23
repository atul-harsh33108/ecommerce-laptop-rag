import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import (fixes deprecation)
from langchain_chroma import Chroma  # Updated import (fixes deprecation)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize embeddings (updated import)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Load vector store (updated import)
vector_store = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings,
    collection_name="laptop_data"
)

# Set up retriever (k=5 most similar chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Initialize Gemini LLM for generation (fixed model name)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Fixed: Updated to current stable model
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Custom prompt template for RAG
prompt_template = """
You are an e-commerce assistant for laptops. Use the following context from product descriptions, reviews, and specs to provide personalized recommendations or comparisons.

Context: {context}

User Query: {question}

Answer concisely, highlighting key specs (e.g., processor, RAM, price, rating) and why it matches the query. If comparing, use a table format. Recommend 2-3 options if possible.
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# Test queries
test_queries = [
    "recommend gaming laptops under $1000",
    "compare thin and light laptops with 16GB RAM"
]

print("=== Basic RAG Pipeline Tests ===\n")
for query in test_queries:
    try:
        result = qa_chain.invoke({"query": query})
        print(f"Query: {query}")
        print(f"Answer: {result['result']}\n")
        print("--- Source Documents (metadata excerpts) ---")
        for doc in result["source_documents"][:2]:  # Show top 2
            print(f"Doc: {doc.metadata['name']} | Price: ${doc.metadata['Price']} | Rating: {doc.metadata['user rating']}")
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error for query '{query}': {str(e)}")
        print("--- Possible fixes: Check API key, wait for quota reset, or switch to local LLM ---\n" + "="*50 + "\n")