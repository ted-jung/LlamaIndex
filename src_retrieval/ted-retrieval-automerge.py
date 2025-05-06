# =============================================================================
# Auto Merging Retriever
# Date: 6, May 2025
# Updated: 6, May 2025
# Writer: Ted, Jung
# Description:
#   Work with HierarchicalNodeParser from a set of leaf nodes
#   Consolidate potentially disparate (from smaller contexts into a larger context)
#   Comparison base retriever and automergingretriever
# =============================================================================



import nest_asyncio

from pathlib import Path

from llama_index.readers.file import PDFReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document

from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
)

from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes

# define storage context
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.llms.openai import OpenAI

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset

nest_asyncio.apply()


llm = OpenAI(model="gpt-4.1-nano")

loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/pdf/llama2.pdf"))


doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]


# Parse Chunk Hierarchy fromm Text, Load into Storage

node_parser = HierarchicalNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(docs)
# print(len(nodes))



# Check if there are children or not. if there are no children, it is a leaf node.

leaf_nodes = get_leaf_nodes(nodes)
# print(leaf_nodes)
# for node in leaf_nodes:
#     print(node)
#     print(node.parent_node)
root_nodes = get_root_nodes(nodes)



# Define docstore and insert nodes into docstore
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

# define storage context (will include vector store by default too)
storage_context = StorageContext.from_defaults(docstore=docstore)




# Load index into vector index
# Comparison base_retriever with automerging_retriever
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
)
base_retriever = base_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

# query_str = "What were some lessons learned from red-teaming?"
# query_str = "Can you tell me about the key concepts for safety finetuning"
query_str = (
    "What could be the potential outcomes of adjusting the amount of safety"
    " data used in the RLHF stage?"
)

base_nodes = base_retriever.retrieve(query_str)
nodes = retriever.retrieve(query_str)