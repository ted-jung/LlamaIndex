# =============================================================================
# Index - PropertyGraph (New way to build knowledge graphs with LLM)
# Created: 4, May 2025
# Updated: 4, May 2025
# Writer: Ted, Jung
# Description:
#   1. for graph instance
#      > docker run \
#       -p 7474:7474 -p 7687:7687 \
#       -v $PWD/data:/data -v $PWD/plugins:/plugins \
#       --name neo4j-apoc \
#       -e NEO4J_apoc_export_file_enabled=true \
#       -e NEO4J_apoc_import_file_enabled=true \
#       -e NEO4J_apoc_import_file_use__neo4j__config=true \
#       -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
#       neo4j:latest 
#   linked together by relationship into structured paths
# =============================================================================


import os
import nest_asyncio


from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
)

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from llama_index.core import Document
from typing import Literal, List, Dict
from pydantic import BaseModel, Field



nest_asyncio.apply()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = OpenAI(model="gpt-4.1-nano")


docs = SimpleDirectoryReader("./data/paul_graham/").load_data()


# Note: used to be `Neo4jPGStore`
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="neo4jneo4j",
    url="bolt://localhost:7687",
)


# a few options (Implicit Extraction, Free-Form Extraction, Schema-guided) 
# example1, Schema-Guided Extraction
entities = ["PERSON", "PLACE", "THING"]
relations = ["PART_OF", "HAS", "IS_A"]
schema = {
    "PERSON": ["PART_OF", "HAS", "IS_A"],
    "PLACE": ["PART_OF", "HAS"], 
    "THING": ["IS_A"],
}

kg_extractors=SchemaLLMPathExtractor(
        llm=llm,
        # possible_entities=entities,
        # possible_relation_props=relations,
        # kg_validation_schema=schema,
        # strict=True,
    )


index = PropertyGraphIndex.from_documents(
    docs,
    kg_extractors=[kg_extractors],
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5"),
    property_graph_store=graph_store,
    show_progress=True,
)


retriever = index.as_retriever(
    include_text=False,  # include source text in returned nodes, default True
)
nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")
for node in nodes:
    print(node.text)


query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("What happened at Interleaf and Viaweb?")
print(str(response))


index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=OpenAI(model="gpt-4o-mini", temperature=0.3),
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5"),
)



document = Document(text="LlamaIndex is great!")

index.insert(document)
nodes = index.as_retriever(include_text=False).retrieve("LlamaIndex")
print(nodes[0].text)