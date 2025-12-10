import os
import json
from supabase import create_client, Client
from flashrank import Ranker, RerankRequest
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
    raise ValueError("Missing environment variables. Please check your .env file.")

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FlashRank (Nano is fast and roughly equivalent to BAAI/bge-reranker-base)
# This downloads the model once (approx 40MB) and caches it.
# You can change cache_dir if needed, or let it default.
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")

def get_embedding(text: str):
    """
    Generates embedding for the query.
    Ensure this matches the model used for your database chunks (e.g., text-embedding-3-small).
    """
    response = openai_client.embeddings.create(
        model="text-embedding-3-small", # CHANGE THIS if you use a different model
        input=text
    )
    return response.data[0].embedding

def retrieve_and_rerank(user_query: str):
    print(f"🎤 Processing Query: {user_query}")
    
    # --- STEP 1: Generate Embedding ---
    try:
        query_vector = get_embedding(user_query)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

    # --- STEP 2: Supabase Hybrid Search ---
    # We fetch more candidates (e.g., 25) to give the reranker enough material to work with.
    rpc_params = {
        "query_text": user_query,
        "query_embedding": query_vector,
        "match_count": 25,       # Fetch top 25 candidates
        "full_text_weight": 1.0, # Balance keyword search
        "semantic_weight": 1.0   # Balance vector search
    }
    
    try:
        response = supabase.rpc("hybrid_search", rpc_params).execute()
    except Exception as e:
        print(f"Error executing Supabase RPC: {e}")
        return []
    
    if not response.data:
        print("No results found in database.")
        return []

    initial_results = response.data
    print(f"🔍 Database found {len(initial_results)} candidates.")

    # --- STEP 3: Re-ranking with FlashRank ---
    # FlashRank needs a list of dictionaries with "id" and "text" (or whatever your content column is)
    # Adjust 'content' key if your table uses a different column name for text
    passages = [
        {"id": str(doc.get("id")), "text": doc.get("content", ""), "meta": doc.get("metadata", {})} 
        for doc in initial_results
    ]

    rerank_request = RerankRequest(query=user_query, passages=passages)
    results = ranker.rerank(rerank_request)

    # --- STEP 4: Return Top Results ---
    # We only take the top 3 for the LLM context to reduce hallucination and latency
    top_results = results[:3]
    
    print(f"🚀 Top 3 after re-ranking:")
    for res in top_results:
        print(f" - [Score: {res['score']:.4f}] {res['text'][:100]}...")
        
    return top_results

# --- Usage Example ---
if __name__ == "__main__":
    # Simulate a voice query
    voice_query = "How do I reset the memory on the T16000M flight stick?"
    
    context_chunks = retrieve_and_rerank(voice_query)
    
    # Construct the final context block for your LLM
    if context_chunks:
        context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
        
        print("\n--- Final Context for LLM ---")
        print(context_text)
    else:
        print("\nNo context retrieved.")
