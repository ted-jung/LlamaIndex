# ===========================================================================
# Ingestion-ingestpipeline(multi-step data processing workflow)
# Created: 21, Feb 2025
# Updated: 23, Feb 2025
# Writer: Ted, Jung
# Description:
#   Ingestion and Cache
# > docker run --name redis-vecdb -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
#   load(from google) -> pipeline[transform(cleaning,splitting,embedding)] 
#                                     -> cache
#                                     -> vectorstore
# ===========================================================================


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.readers.google import GoogleDriveReader

from redisvl.schema import IndexSchema


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

custom_schema = IndexSchema.from_dict(
    {
        "index": {"name": "gdrive", "prefix": "doc"},
        # customize fields that are indexed
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # custom vector field for bge-small-en-v1.5 embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    }
)



# Define Vector Store (clear vector store if exists)
vector_store = RedisVectorStore(
    schema=custom_schema,
    redis_url="redis://localhost:6379",
)

if vector_store.index_exists():
    vector_store.delete_index()



# Set up the ingestion cache layer
cache = IngestionCache(
    cache=RedisCache.from_host_and_port("localhost", 6379),
    collection="redis_cache",
)


# Define Ingestion Pipeline (sentence splitting, embedding transformations)
# - document: storing and managing document metadata,content
# - vector_store: storeing and managing document embedding
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    docstore=RedisDocumentStore.from_host_and_port(
        "localhost", 6379, namespace="document_store"
    ),
    vector_store=vector_store,
    cache=cache,
    docstore_strategy=DocstoreStrategy.UPSERTS,
)



# Define vector store index
index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, embed_model=embed_model
)



# Load data from GoogleDrive

loader = GoogleDriveReader()
def load_data(folder_id: str):
    docs = loader.load_data(folder_id=folder_id)
    for doc in docs:
        doc.id_ = doc.metadata["file_name"]
    return docs

docs = load_data(folder_id="ted-xxxxxxxxx")
nodes = pipeline.run(documents=docs)



# Ask questions over initial data
query_engine = index.as_query_engine()
response = query_engine.query("What are the sub-types of question answering?")
print(str(response))


# Modify and Reload the Data
docs = load_data(folder_id="ted-xxxxxxxx")
nodes = pipeline.run(documents=docs)
print(f"Ingested {len(nodes)} Nodes")


response = query_engine.query("What are the sub-types of question answering?")
print(str(response))
