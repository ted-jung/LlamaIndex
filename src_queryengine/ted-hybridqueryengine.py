


import asyncio
import os
import chromadb

from llama_index.core import (
    Settings,
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    KnowledgeGraphIndex,
    PropertyGraphIndex
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolOutput
from llama_index.core.schema import TextNode, Document, Node



# Load data (replace with your actual data loading)
reader = SimpleDirectoryReader(input_files=["./data/paul_graham/paul_graham_essay.txt"])
documents = reader.load_data()



# Initialize LLM and Embeddings
llm = OpenAI(model="gpt-4.1-nano")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model


# --- 1. Create and Populate Vector Store Index ---
# Define Chroma vector store
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

chroma_persist_dir = "./chroma_hybrid"
if not os.path.exists(chroma_persist_dir):
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context_vector = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context_vector, 
        embed_model=embed_model
    )
    vector_index.storage_context.persist()
else:
    storage_context_vector = StorageContext.from_defaults(persist_dir=chroma_persist_dir)
    vector_index = load_index_from_storage(storage_context_vector, embed_model=embed_model)



# --- 2. Create and Populate Graph Store Index ---
# Assuming you have a way to extract entities and relationships from your documents
# For this example, we'll create some dummy graph data based on the essay content
graph_nodes = [
    Node(id_="paul_graham", metadata={"name": "Paul Graham", "type": "PERSON"}),
    Node(id_="ycombinator", metadata={"name": "Y Combinator", "type": "ORGANIZATION"}),
    Node(id_="essays", metadata={"name": "Essays", "type": "TOPIC"}),
    Node(id_="startups", metadata={"name": "Startups", "type": "TOPIC"}),
]
graph_edges = [
    {"source": "paul_graham", "target": "ycombinator", "relation": "FOUNDED"},
    {"source": "paul_graham", "target": "essays", "relation": "WROTE_ABOUT"},
    {"source": "ycombinator", "target": "startups", "relation": "FOCUSES_ON"},
    {"source": "essays", "target": "startups", "relation": "DISCUSSES"},
]


# Initialize NebulaGraphStore (replace with your actual connection details)
# This is a placeholder; you'll need to set up a NebulaGraph instance
space_name = "paul_graham_essay"
edge_types, rel_prop_names = ["relationship"], ["relationship"]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg


os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"  # replace with your password, by default it is "nebula"
os.environ["NEBULA_ADDRESS"] = "127.0.0.1:9669" 

try:
    graph_store = NebulaPropertyGraphStore(
        space_name=space_name,
        username=os.environ["NEBULA_USER"],
        password=os.environ["NEBULA_PASSWORD"],
        url=os.environ["NEBULA_ADDRESS"],
    )

    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
    # graph_index = KnowledgeGraphIndex.from_documents(
    #     documents,
    #     storage_context=storage_context_graph,
    #     max_triplets_per_chunk=2,
    #     space_name=space_name,
    #     edge_types=edge_types,
    #     rel_prop_names=rel_prop_names,
    #     tags=tags,
    #     llm=llm, 
    # )

    # graph_index = PropertyGraphIndex(
    #     nodes=graph_nodes,
    #     edges=graph_edges,
    #     storage_context=storage_context_graph,
    #     llm=llm,
    # )

    graph_index = PropertyGraphIndex.from_documents(
        documents,
        # storage_context=storage_context,
    )

    graph_index.storage_context.persist()


except ImportError:
    print("NebulaGraph not installed. Skipping graph index creation.")
    graph_index = None
except Exception as e:
    print(f"Error creating NebulaGraph index: {e}")
    graph_index = None



# --- 3. Create Query Engines and Tools ---
vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)
vector_tool = QueryEngineTool(
    query_engine=vector_query_engine,
    metadata="Useful for answering questions about the content of the essays based on semantic similarity.",
)

graph_query_engine = None
graph_tool = None
if graph_index:
    graph_query_engine = graph_index.as_query_engine(
        llm = llm
        # You might need to configure graph-specific query parameters
    )
    graph_tool = QueryEngineTool(
        query_engine=graph_query_engine,
        metadata="Useful for answering questions about relationships and connections between entities in the essays.",
    )



# --- 4. Create a SubQuestionQueryEngine for Hybrid Search ---
query_engine_tools = [vector_tool]
if graph_tool:
    query_engine_tools.append(graph_tool)

hybrid_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    llm=llm,
    verbose=True
)



# --- 5. Perform Hybrid Search ---
async def main():
    query = "What did Paul Graham found and what topics did his essays discuss?"
    response = await hybrid_query_engine.aquery(query)
    print(f"Hybrid Query Response: {response}")

    query = "Find essays that discuss startups and mention Y Combinator."
    response = await hybrid_query_engine.aquery(query)
    print(f"Hybrid Query Response: {response}")

    if graph_index:
        query = "What kind of organization is Y Combinator and who founded it?"
        response = await hybrid_query_engine.aquery(query)
        print(f"Hybrid Query Response (Graph focus): {response}")


asyncio.run(main)