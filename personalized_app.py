import streamlit as st
import pandas as pd
import time
import os  # <-- Added missing import for os.getenv
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Page config for better look
st.set_page_config(page_title="Laptop Recommender RAG", page_icon="🛒", layout="wide")

# Initialize embeddings and vector store
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings,
        collection_name="laptop_data"
    )
    return vector_store, embeddings

vector_store, embeddings = load_vector_store()

# Initialize Gemini LLM
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Using 1.5-flash as 2.5 is not generally available
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

llm = load_llm()

# Prompt template
prompt_template = """
You are an e-commerce assistant for laptops. Use the following positive-review context to provide personalized recommendations based on user preferences in the query.

Context: {context}

User Query: {question}

Answer concisely, highlighting key specs (processor, RAM, price, rating, sentiment) and why it matches preferences. Recommend 2-3 options in a table. Simulate adaptation from past interactions.
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Function to build dynamic filter LIST from user prefs
def build_filter_list(budget_max, min_ram, preferred_brand, min_rating, min_screen, max_screen, preferred_gpu, preferred_os, max_weight, min_ssd):
    filter_list = []
    
    if budget_max and budget_max > 0:
        filter_list.append({"Price": {"$lte": budget_max}})
        
    if min_ram and min_ram > 0:
        filter_list.append({"RAM": {"$gte": min_ram}})
        
    if preferred_brand:
        brands = [b.strip() for b in preferred_brand.split(",") if b.strip()]
        if brands:
            # Assuming 'Processor Brand' is the metadata field. Adjust if it's 'Brand' or 'name'
            # Let's check a few common fields. Chroma filtering can be strict.
            # We'll assume the field is 'Processor Brand' as in your code.
            filter_list.append({"Processor Brand": {"$in": brands}}) 
            
    if min_rating and min_rating > 0:
        filter_list.append({"user rating": {"$gte": min_rating}})
        
    if min_screen and min_screen > 0:
        filter_list.append({"Screen Size": {"$gte": min_screen}})
        
    if max_screen and max_screen > 0:
        filter_list.append({"Screen Size": {"$lte": max_screen}})
        
    if preferred_gpu:
        gpus = [g.strip() for g in preferred_gpu.split(",") if g.strip()]
        if gpus:
            filter_list.append({"Graphic Processor": {"$in": gpus}})
            
    if preferred_os:
        oss = [o.strip() for o in preferred_os.split(",") if o.strip()]
        if oss:
            filter_list.append({"Operating System": {"$in": oss}})
            
    if max_weight and max_weight > 0:
        filter_list.append({"Weight": {"$lte": max_weight}})
        
    if min_ssd and min_ssd > 0:
        filter_list.append({"SSD Capacity": {"$gte": min_ssd}})
            
    return filter_list

# Streamlit UI
st.title("🛒 Personalized Laptop Recommender RAG")
st.caption("Powered by LangChain, Chroma, Gemini, & HuggingFace | Dataset: 984 Laptops")

# Session state for "learning"
if "prefs_history" not in st.session_state:
    st.session_state.prefs_history = []
if "recs_history" not in st.session_state:
    st.session_state.recs_history = []

# Sidebar for prefs
st.sidebar.header("🔧 User Preferences")
col1, col2 = st.sidebar.columns(2)
budget_max = col1.number_input("Max Budget ($)", min_value=0.0, value=0.0, step=100.0)  # Default null (0)
min_ram = col1.number_input("Min RAM (GB)", min_value=0, value=0, step=4)  # Default null (0)
preferred_brand = st.sidebar.text_input("Preferred Brands (comma-separated, e.g., Intel,AMD)", value="")  # Default empty
min_rating = col2.slider("Min Rating", min_value=0.0, max_value=5.0, value=0.0, step=0.5)  # Default null (0)
num_recs = col2.slider("Number of Recs", min_value=3, max_value=10, value=5, step=1)

# Additional preferences
min_screen = col1.number_input("Min Screen Size (inches)", min_value=0.0, value=0.0, step=1.0)  # Default null
max_screen = col1.number_input("Max Screen Size (inches)", min_value=0.0, value=0.0, step=1.0)  # Default null
preferred_gpu = st.sidebar.text_input("Preferred GPU (comma-separated, e.g., NVIDIA,Integrated)", value="")  # Default empty
preferred_os = st.sidebar.text_input("Preferred OS (comma-separated, e.g., Windows 11,Windows 10)", value="")  # Default empty
max_weight = col2.number_input("Max Weight (kg)", min_value=0.0, value=0.0, step=0.5)  # Default null
min_ssd = col2.number_input("Min SSD (GB)", min_value=0, value=0, step=128)  # Default null

query = st.text_input("Your Query (e.g., 'recommend gaming laptops')", value="recommend gaming laptops")

# History expander
with st.sidebar.expander("📜 Past Recommendations", expanded=False):
    # Iterate in reverse to show newest first
    for i, rec in enumerate(reversed(st.session_state.recs_history[-3:])):
        st.write(f"**Query {len(st.session_state.recs_history)-i}:** {rec['query']}")
        if 'sources' in rec and rec['sources']: # Check if sources list is not empty
            st.write("Sources:", [s['name'][:50] + "..." for s in rec['sources'][:2]])
        else:
            st.write("Sources: (None found)")


# Main content
col_left, col_right = st.columns([3, 1])

# =================================================================
# START: UPDATED CODE BLOCK WITH FIX
# =================================================================
with col_left:
    if st.button("🚀 Get Personalized Recommendations", use_container_width=True):
        start_time = time.time()
        if query:
            # Build final filter
            final_filter_list = [{"sentiment_score": {"$gte": 0.5}}]
            user_filters = build_filter_list(budget_max, min_ram, preferred_brand, min_rating, min_screen, max_screen, preferred_gpu, preferred_os, max_weight, min_ssd)
            if user_filters:
                final_filter_list.extend(user_filters)
            
            if len(final_filter_list) > 1:
                final_filter = {"$and": final_filter_list}
            elif final_filter_list:
                final_filter = final_filter_list[0]
            else:
                final_filter = None

            # Retriever with num_recs (k)
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": num_recs,
                    "filter": final_filter
                }
            )
            
            # Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            # Inject prefs into query
            prefs_summary = f" (Budget: ${budget_max}, Min RAM: {min_ram}GB, Brands: {preferred_brand or 'Any'}, Min Rating: {min_rating}, Min Screen: {min_screen}\", Max Screen: {max_screen}\", GPU: {preferred_gpu or 'Any'}, OS: {preferred_os or 'Any'}, Max Weight: {max_weight}kg, Min SSD: {min_ssd}GB)"
            full_query = query + prefs_summary
            
            try:
                result = qa_chain.invoke({"query": full_query})
            except Exception as e:
                st.error(f"An error occurred while running the RAG chain: {e}")
                st.stop() # Stop execution if the chain fails
            
            # Metrics
            latency = time.time() - start_time
            num_sources = len(result["source_documents"]) # Get the count
            
            # Display metrics in columns
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Query Latency", f"{latency:.2f}s")
            metric_col2.metric("Sources Retrieved", num_sources)
            
            # Recommendations
            st.subheader("📋 Recommendations")
            st.write(result['result']) # Show LLM result regardless
            
            # --- START OF THE FIX ---
            # Only try to build charts/tables IF we found sources
            if num_sources > 0:
                # Visual: Price/RAM bar chart from sources
                source_data = []
                for doc in result["source_documents"]:
                    source_data.append({
                        'Name': doc.metadata.get('name', 'Unknown')[:30] + "...", # Use .get for safety
                        'Price': doc.metadata.get('Price', 0),
                        'RAM': doc.metadata.get('RAM', 0)
                    })
                df_sources = pd.DataFrame(source_data)
                
                # This code is now safe because df_sources is not empty
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("💰 Price Distribution")
                    st.bar_chart(df_sources.set_index('Name')['Price'])
                with col2:
                    st.subheader("💾 RAM Distribution")
                    st.bar_chart(df_sources.set_index('Name')['RAM'])
                
                # Sources table
                st.subheader("📄 Source Documents")
                display_data = []
                for doc in result["source_documents"]:
                    score = doc.metadata.get('sentiment_score', 'N/A')
                    display_data.append({
                        'Laptop': doc.metadata.get('name', 'Unknown')[:50] + "...",
                        'Price ($)': doc.metadata.get('Price', 'N/A'),
                        'RAM (GB)': doc.metadata.get('RAM', 'N/A'),
                        'Rating': doc.metadata.get('user rating', 'N/A'),
                        'Sentiment': f"{score:.2f}" if isinstance(score, (int, float)) else score
                    })
                df_display = pd.DataFrame(display_data)
                st.dataframe(df_display, use_container_width=True)
                
                # Export
                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Recs as CSV",
                    data=csv,
                    file_name=f"laptop_recs_{int(time.time())}.csv",
                    mime='text/csv'
                )
                
                # Store history
                st.session_state.prefs_history.append(prefs_summary)
                st.session_state.recs_history.append({
                    'query': query,
                    'sources': [{'name': doc.metadata.get('name', 'Unknown')} for doc in result["source_documents"]]
                })
            
            else:
                # Show a warning if no sources were found
                st.warning("⚠️ No laptops found matching your exact criteria. Please try broadening your search filters (e.g., increase budget or lower minimum RAM).")
                # Still store the preference summary
                st.session_state.prefs_history.append(prefs_summary)
                # Store the query with empty sources
                st.session_state.recs_history.append({
                    'query': query,
                    'sources': []
                })
            # --- END OF THE FIX ---
                
        else:
            st.warning("Enter a query to get recommendations!")
# =================================================================
# END: UPDATED CODE BLOCK
# =================================================================


# Comparison Tool
with st.expander("⚖️ Quick Product Comparison", expanded=False):
    st.info("Select 2 laptops to compare specs.")
    try:
        # Fetching a larger set to increase chances, but still limiting for performance
        metas = vector_store.get(limit=100)['metadatas'] 
        laptop_names = sorted(list(set([meta['name'] for meta in metas if 'name' in meta]))) # Get unique names
        if not laptop_names:
            st.warning("No laptop names found in vector store metadata.")
    except Exception as e:
        st.error(f"Error loading laptops: {e}")
        laptop_names = []

    if laptop_names: # Only show selectors if we have names
        col1, col2 = st.columns(2)
        lap1 = col1.selectbox("Laptop 1", laptop_names, index=0)
        # Ensure default for lap2 is different if possible
        lap2_index = 1 if len(laptop_names) > 1 else 0
        lap2 = col2.selectbox("Laptop 2", laptop_names, index=lap2_index)
        
        if st.button("Compare"):
            if lap1 == lap2:
                st.warning("Please select two different laptops to compare.")
            else:
                # Use a more reliable way to get specific docs by metadata filter
                docs1 = vector_store.get(where={"name": lap1}, limit=1)['documents']
                meta1 = vector_store.get(where={"name": lap1}, limit=1)['metadatas']
                docs2 = vector_store.get(where={"name": lap2}, limit=1)['documents']
                meta2 = vector_store.get(where={"name": lap2}, limit=1)['metadatas']

                if meta1 and meta2:
                    m1 = meta1[0] # Get the first (and only) metadata dict
                    m2 = meta2[0]
                    
                    # Formatting sentiment score
                    s1 = m1.get('sentiment_score', 'N/A')
                    s2 = m2.get('sentiment_score', 'N/A')
                    s1_formatted = f"{s1:.2f}" if isinstance(s1, (int, float)) else s1
                    s2_formatted = f"{s2:.2f}" if isinstance(s2, (int, float)) else s2

                    comp_data = {
                        'Spec': ['Price ($)', 'RAM (GB)', 'Processor', 'Rating', 'Sentiment'],
                        lap1[:30] + "...": [m1.get('Price', 'N/A'), m1.get('RAM', 'N/A'), m1.get('Processor Name', 'N/A'), m1.get('user rating', 'N/A'), s1_formatted],
                        lap2[:30] + "...": [m2.get('Price', 'N/A'), m2.get('RAM', 'N/A'), m2.get('Processor Name', 'N/A'), m2.get('user rating', 'N/A'), s2_formatted]
                    }
                    st.table(pd.DataFrame(comp_data).set_index('Spec'))
                else:
                    st.warning("Could not retrieve metadata for one or both laptops.")
    else:
        st.info("No laptops available for comparison.")


# Footer
st.caption("Upgrade your shopping with AI-powered personalization! 🎯")