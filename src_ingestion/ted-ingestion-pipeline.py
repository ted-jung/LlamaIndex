# ===========================================================================
# Ingestion-ingestpipeline(multi-step data processing workflow)
# Created: 21, Feb 2025
# Updated: 22, Feb 2025
# Writer: Ted, Jung
# Description: 
#   Ingestion and Cache
#   use docker run
#   > docker run --name redis-vecdb -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
#   load -> transform(cleaning,splitting,embedding) -> cache -> vectorstore
# ===========================================================================


import os
import re
import clickhouse_connect

from llama_index.core import Document,Settings
from llama_index.core.schema import TransformComponent
from llama_index.core import SimpleDirectoryReader

from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.ingestion import IngestionCache
from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.openai import OpenAI


Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")



# Create a client & VectorStore(ClickHouse)

ch_client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    username="default",
    password="magic",
    database="default",
)
vector_store = ClickHouseVectorStore(
    ch_client, 
    table="quickstart_index",
    embed_model=embed_model
)



# Define text_splitter & the textcleaner to be used when transforming In IngestPipeline

text_splitter = TokenTextSplitter(chunk_size=512)
class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        modified_nodes = []
        for node in nodes:
            modified_text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
            new_node = Document(text=modified_text, doc_id=node.doc_id, metadata=node.metadata)
            modified_nodes.append(new_node)
        return modified_nodes
    


# Integrates well with IngestPipeline
# : caching of intermediate results during the ingestion process
ingest_cache = IngestionCache(
    cache=RedisCache.from_host_and_port(host="127.0.0.1", port=6379),
    collection="my_test_cache",
)

pipeline = IngestionPipeline(
    transformations=[
        TextCleaner(),
        text_splitter,
        embed_model,
    ],
    vector_store=vector_store,
    cache=ingest_cache, 
)

curr_dir = os.getcwd()
documents = SimpleDirectoryReader(f"{curr_dir}/data/paul_graham/").load_data()
nodes = pipeline.run(documents=documents)




index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

query_engine = index.as_query_engine()

print(query_engine.query("What did the author do growing up? especially regarding IBM"))

print(100*"=")

pipeline = IngestionPipeline(
    transformations=[TextCleaner(), text_splitter, embed_model],
    cache=ingest_cache,
)

nodes = pipeline.run(documents=documents)

# ingest_cache.clear()

# print(100*"=")

# pipeline = IngestionPipeline(
#     transformations=[TextCleaner(), text_splitter, embed_model],
#     cache=ingest_cache,
# )

# nodes = pipeline.run(documents=documents)
