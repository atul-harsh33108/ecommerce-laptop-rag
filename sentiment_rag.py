import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document  # For updates
from transformers import pipeline

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Load vector store
vector_store = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings,
    collection_name="laptop_data"
)

# Initialize sentiment analysis pipeline (local HF model) with truncation
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # CPU
    truncation=True,  # Truncate long texts to max_length
    max_length=512
)

# Function to get sentiment score (positive: >0.5, negative: <0.5)
def get_sentiment_score(text):
    result = sentiment_pipeline(text)[0]
    score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
    return score

# Add sentiment scores to vector store metadata (one-time; skip if already present)
print("Checking/Adding sentiment scores to vector store...")
docs_dict = vector_store.get()
existing_scores = any('sentiment_score' in meta for meta in docs_dict['metadatas'])
if not existing_scores:
    print("Adding sentiment scores...")
    
    ids_to_update = []
    docs_to_update = []
    
    for i in range(len(docs_dict['documents'])):
        content = docs_dict['documents'][i]
        current_id = docs_dict['ids'][i]
        current_metadata = docs_dict['metadatas'][i]
        
        # 1. Get the score
        score = get_sentiment_score(content)
        
        # 2. Create the new metadata
        updated_metadata = current_metadata.copy()
        updated_metadata['sentiment_score'] = score
        
        # 3. Create the updated Document object
        updated_doc = Document(
            page_content=content,  # Content must be provided
            metadata=updated_metadata
        )
        
        # 4. Append to their respective lists
        ids_to_update.append(current_id)
        docs_to_update.append(updated_doc)
        
    # Batch update using the correct two arguments
    if ids_to_update:
        vector_store.update_documents(
            ids=ids_to_update, 
            documents=docs_to_update
        )
        print(f"Sentiment scores added/updated for {len(ids_to_update)} documents!")
    else:
        print("No documents to update.")
        
else:
    print("Sentiment scores already present—skipping.")

# Set up sentiment-filtered retriever (score > 0.5 for positive bias)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5, "filter": {"sentiment_score": {"$gte": 0.5}}}
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Updated prompt to mention sentiment
prompt_template = """
You are an e-commerce assistant for laptops. Use the following positive-review context from product descriptions, reviews, and specs to provide personalized recommendations or comparisons.

Context: {context}

User Query: {question}

Answer concisely, highlighting key specs (e.g., processor, RAM, price, rating, sentiment) and why it matches the query. If comparing, use a table format. Recommend 2-3 options if possible.
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create updated RAG chain
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

print("=== Sentiment-Integrated RAG Tests ===\n")
for query in test_queries:
    try:
        result = qa_chain.invoke({"query": query})
        print(f"Query: {query}")
        print(f"Answer: {result['result']}\n")
        print("--- Source Documents (with sentiment) ---")
        for doc in result["source_documents"][:2]:
            score = doc.metadata.get('sentiment_score', 'N/A')
            print(f"Doc: {doc.metadata['name']} | Price: ${doc.metadata['Price']} | Rating: {doc.metadata['user rating']} | Sentiment: {score:.2f}")
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error for query '{query}': {str(e)}\n" + "="*50 + "\n")