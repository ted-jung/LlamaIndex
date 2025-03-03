# ===========================================================================
# LlamaParse
# Created: 2, Mar 2025
# Updated: 2, Mar 2025
# Writer: Ted, Jung
# Description: 
#   LlamaParse -> splitter(document, nodes) -> postprocessor -> engine
# ===========================================================================



import os

# llama-parse is async-first, running the async code in a notebook requires the use of nest_asyncio
import nest_asyncio


from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


nest_asyncio.apply()


# API access to llama-cloud
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..."


embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini")

Settings.llm = llm
Settings.embed_model = embed_model


# LlamaParse PDF reader for PDF Parsing
curr_dir = os.getcwd()
documents = LlamaParse(result_type="markdown").load_data(
    f"{curr_dir}/data/pdf/uber_10q_march_2022.pdf"
)

# print(documents[0].text[:1000] + "...")


# This is good for document having heading, table and paragraphs
# Turn documents to Nodes
# granular parsing
node_parser = MarkdownElementNodeParser(
    llm=OpenAI(model="gpt-4o-mini"), num_workers=8
)


# Two methods 
# : documents -> a list of nodes
# : nodes -> nodes(text, index)
# index nodes are very useful for recursive retrieval

nodes = node_parser.get_nodes_from_documents(documents)
text_nodes, index_nodes = node_parser.get_nodes_and_objects(nodes)

# print(text_nodes[0])
# print(index_nodes[0])

recursive_index = VectorStoreIndex(nodes=text_nodes + index_nodes)
raw_index = VectorStoreIndex.from_documents(documents)   # it uses SimpleNodeParser



# reranking, provide accurate relevance scoring
# Improving the overall effectiveness of RAG systems 
# by ensuring that the language model works with the most relevant context.
reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
)

recursive_query_engine = recursive_index.as_query_engine(
    similarity_top_k=15, node_postprocessors=[reranker], verbose=True
)
raw_query_engine = raw_index.as_query_engine(
    similarity_top_k=15, node_postprocessors=[reranker]
)


query = "What is the change of free cash flow and what is the rate from the financial and operational highlights?"

response_1 = raw_query_engine.query(query)
print("\n************New LlamaParse+ Basic Query Engine************")
print(response_1)

response_2 = recursive_query_engine.query(query)
print(
    "\n************New LlamaParse+ Recursive Retriever Query Engine************"
)
print(response_2)

