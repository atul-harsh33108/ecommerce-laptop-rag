# import pandas as pd
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# # Initialize Gemini embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

# # Load cleaned dataset
# df = pd.read_csv('data/cleaned_laptop_data.csv')

# # Combine text fields for chunking
# df['text_to_embed'] = df['Additional Features'] + " " + df['simulated_reviews']

# # Initialize text splitter
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=512,
#     chunk_overlap=50,
#     length_function=len
# )

# # Prepare documents and metadata
# documents = []
# metadatas = []
# ids = []

# for idx, row in df.iterrows():
#     # Split text into chunks
#     chunks = text_splitter.split_text(row['text_to_embed'])
    
#     # Create metadata for each chunk
#     metadata = {
#         'name': row['name'],
#         'Price': row['Price'],
#         'Processor Brand': row['Processor Brand'],
#         'Processor Name': row['Processor Name'],
#         'RAM': row['RAM'],
#         'SSD Capacity': row['SSD Capacity'] if pd.notna(row['SSD Capacity']) else 'None',
#         'Graphic Processor': row['Graphic Processor'] if pd.notna(row['Graphic Processor']) else 'Integrated',
#         'Screen Size': row['Screen Size'],
#         'Screen Resolution': row['Screen Resolution'],
#         'user rating': row['user rating']
#     }
    
#     # Add each chunk with its metadata
#     for chunk_idx, chunk in enumerate(chunks):
#         documents.append(chunk)
#         metadatas.append(metadata)
#         ids.append(f"{idx}_{chunk_idx}")

# # Create Chroma vector store
# vector_store = Chroma.from_texts(
#     texts=documents,
#     embedding=embeddings,
#     metadatas=metadatas,
#     ids=ids,
#     collection_name="laptop_data",
#     persist_directory="data/chroma_db"
# )

# # Persist the vector store
# vector_store.persist()

# # Print summary
# print(f"Number of chunks: {len(documents)}")
# print(f"Sample document: {documents[0]}")
# print(f"Sample metadata: {metadatas[0]}")
# print(f"Sample ID: {ids[0]}")
# print("Vector store created and persisted at data/chroma_db")



import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Initialize local HuggingFace embeddings (downloads model on first run)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU
)

# Load cleaned dataset
df = pd.read_csv('data/cleaned_laptop_data.csv')

# Combine text fields for chunking
df['text_to_embed'] = df['Additional Features'] + " " + df['simulated_reviews']

# Initialize text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=512,
    chunk_overlap=50,
    length_function=len
)

# Prepare documents and metadata
documents = []
metadatas = []
ids = []

for idx, row in df.iterrows():
    # Split text into chunks
    chunks = text_splitter.split_text(row['text_to_embed'])
    
    # Create metadata for each chunk
    metadata = {
        'name': row['name'],
        'Price': row['Price'],
        'Processor Brand': row['Processor Brand'],
        'Processor Name': row['Processor Name'],
        'RAM': row['RAM'],
        'SSD Capacity': row['SSD Capacity'] if pd.notna(row['SSD Capacity']) else 'None',
        'Graphic Processor': row['Graphic Processor'] if pd.notna(row['Graphic Processor']) else 'Integrated',
        'Screen Size': row['Screen Size'],
        'Screen Resolution': row['Screen Resolution'],
        'user rating': row['user rating']
    }
    
    # Add each chunk with its metadata
    for chunk_idx, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append(metadata)
        ids.append(f"{idx}_{chunk_idx}")

# Create Chroma vector store
vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    metadatas=metadatas,
    ids=ids,
    collection_name="laptop_data",
    persist_directory="data/chroma_db"
)

# Persist the vector store
vector_store.persist()

# Print summary
print(f"Number of chunks: {len(documents)}")
print(f"Sample document: {documents[0][:200]}...")  # Truncated for readability
print(f"Sample metadata keys: {list(metadatas[0].keys())}")
print(f"Sample ID: {ids[0]}")
print("Vector store created and persisted at data/chroma_db")