# =============================================================================
# Title: How to search meta(original text) from embedded value
# Created: 18, Jun 2025
# Updated: 18, Jun 2025
# Writer: Ted, Jung
# Description:
#   1. Create table first
#       CREATE or REPLACE TABLE doc_embeddings
#       (
#       	id UUID DEFAULT generateUUIDv4(),
#           doc_id UUID DEFAULT generateUUIDv4(),
#           text String,
#           vector Array(Float32)
#       )
#       ENGINE = MergeTree
#       ORDER BY id;
#   2. Insert using the library of clickhouse-client
#   3. Do a similarity search with embedded value (sentence embedding search)
# =============================================================================


import uuid
import numpy as np
import clickhouse_connect


from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.clickhouse import ClickHouseVectorStore



# Model(Embedding, LLM)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.embed_model = embed_model
Settings.llm = OpenAI(model="gpt-4.1-nano", request_timeout=360.0)



# Text you want to store
text = "LlamaIndex makes it easy to connect your LLM to data."
reader = SimpleDirectoryReader(input_dir="./data/paul_graham/")
documents = reader.load_data()
index = VectorStoreIndex.from_documents(documents=documents)
ted_engine = index.as_query_engine()
ted_response = ted_engine.query("who is paul graham")

print(f"reponse: {ted_response}")
# Embed it
embedding = embed_model.get_text_embedding(text)



# Connect to ClickHouse
ch_client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    username="default",
    password="magic",
    database="default",
)


# Option-1 ====================================================================
# Insert text and embedding with metadata
# record_id = str(uuid.uuid4())  # Generate a UUID
# record_id2 = str(uuid.uuid4())  # Generate a UUID


for doc in documents:
    record_id1 = str(uuid.uuid4())  # Generate a UUID
    record_id2 = str(uuid.uuid4())  # Generate a UUID

    embedding = embed_model.get_text_embedding(doc.text)
    response = ch_client.insert(
        table='doc_embeddings',
        column_names=['id', 'doc_id', 'text', 'vector'],
        data=[[record_id1, record_id2, doc.text, embedding]]
    )

print(response)

result = ch_client.query('SELECT * FROM doc_embeddings')
for row in result.result_rows:
    print(row[2])

# Similarity search
query = "What is paul graham doing now a days?"
query_vector = embed_model.get_text_embedding(query)

query_vector = np.array(query_vector, dtype=np.float32).tolist()

vector_str = "[" + ", ".join([str(v) for v in query_vector]) + "]"

query = f"""
SELECT
    id,
    doc_id,
    text,
    cosineDistance(vector, {vector_str}) AS distance
FROM doc_embeddings
ORDER BY distance ASC
LIMIT 5
"""

nearest_neighbors = ch_client.query(query)
rows = nearest_neighbors.result_rows

for row in rows:
    print(f"==========================Nearest text: {row[2][:200]}")





# Option-2 ====================================================================
# vector_store = ClickHouseVectorStore(
#     ch_client, 
#     table="doc_embeddings",
#     embed_model=embed_model
# )
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# print("haha")