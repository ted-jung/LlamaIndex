# This is a conceptual snippet.
# You would need to install llama-index and a vector store client (e.g., pinecone-client)
# pip install llama-index openai pinecone-client sqlalchemy

import os
import pandas as pd

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from sqlalchemy import create_engine, text


# --- Configuration ---
# Set your OpenAI/Pinecone API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"

PINECONE_INDEX_NAME = "tour-planning-hotels"
DB_CONNECTION_STRING = "postgresql://user:password@host:port/database" # Replace with your DB connection



# --- 1. Data Loading and Preprocessing (Simulated from your DB) ---

def load_and_preprocess_hotels_from_db():
    """
    Simulates loading hotel data from your relational database and concatenating fields.
    In a real app, you'd fetch this from your DB.
    """
    engine = create_engine(DB_CONNECTION_STRING)
    # Example: Fetching hotels data
    # In a real scenario, you'd fetch all relevant columns
    query = "SELECT hotel_id, hotel_name, description, location, star_rating, amenities, price_range FROM hotels"
    df = pd.read_sql(query, engine)

    documents = []
    for index, row in df.iterrows():
        # Concatenate relevant fields into a single string for embedding
        content = (
            f"Hotel Name: {row['hotel_name']}. "
            f"Description: {row['description']}. "
            f"Location: {row['location']}. "
            f"Star Rating: {row['star_rating']} stars. "
            f"Amenities: {row['amenities']}. "
            f"Price Range: {row['price_range']}."
        )
        # Store original hotel_id as metadata for retrieval
        doc = Document(
            text=content,
            metadata={"hotel_id": row['hotel_id'], "hotel_name": row['hotel_name'], "location": row['location']}
        )
        documents.append(doc)
    return documents



# --- 2. Initialize LlamaIndex Components ---

# Initialize LLM and Embedding Model and Pinecone
llm = OpenAI(model="gpt-4.1-nano") # Or gpt-3.5-turbo
embed_model = OpenAIEmbedding(model="text-embedding-ada-002") # Or text-embedding-3-small/large
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))



# --- 3. Create/Load Vector Store Index ---

def setup_hotel_index():
    # Check if index exists, if not, create it
    if PINECONE_INDEX_NAME not in pc.list_indexes().names:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536, # Dimension for text-embedding-ada-002
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-west-2') # Or your preferred cloud/region
        )

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Load documents from your database
    documents = load_and_preprocess_hotels_from_db()

    # Create index from documents and store in Pinecone
    # This step will generate embeddings and upload them to Pinecone
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        embed_model=embed_model,
        llm=llm # LLM is used for node parsing and other internal operations
    )
    return index

 # --- 4. Query the Index ---

def query_hotel_index(query_text: str, index: VectorStoreIndex):
        # Create a query engine
    query_engine = index.as_query_engine(
        similarity_top_k=3, # Retrieve top 3 most similar hotels
        response_mode="tree_summarize" # Summarize retrieved nodes
    )
    response = query_engine.query(query_text)
    return response


# --- Main Execution ---
if __name__ == "__main__":
    print("Setting up LlamaIndex for hotels data...")
    hotel_index = setup_hotel_index()
    
    print(f"LlamaIndex for hotels '{PINECONE_INDEX_NAME}' ready.")

    print("\nQuerying for romantic hotels in Paris with a spa:")
    query_result = query_hotel_index("romantic hotels in Paris with a spa", hotel_index)
    print(query_result)

    print("\nQuerying for budget-friendly hotels in New York:")
    query_result = query_hotel_index("budget-friendly hotels in New York", hotel_index)
    print(query_result)

    # You can also access the source nodes to get metadata like hotel_id
    # for further SQL lookups in your backend
    # for node in query_result.source_nodes:
    #     print(f"Retrieved Hotel ID: {node.metadata.get('hotel_id')}")
    #     print(f"Content: {node.text[:100]}...") # Print first 100 chars of content