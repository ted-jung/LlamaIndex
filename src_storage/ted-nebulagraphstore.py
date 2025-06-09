
import os
import logging
import sys

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.openai import OpenAI
from IPython.display import Markdown, display


llm = OpenAI(temperature=0, model="gpt-4.1-nano")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.chunk_size = 512


documents = SimpleDirectoryReader(input_files=["./data/paul_graham/paul_graham_essay.txt"]).load_data()


space_name = "paul_graham_essay"
edge_types, rel_prop_names = ["relationship"], ["relationship"]
tags = ["entity"]


graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

storage_context.persist()

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

query_engine = index.as_query_engine()

response = query_engine.query("Tell me more about Interleaf")

print(response)