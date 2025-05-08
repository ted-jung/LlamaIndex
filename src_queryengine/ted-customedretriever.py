# =============================================================================
# Hybrid retrieve
# Date: 13, Jan 2025
# Updated: 8, May 2025
# Writer: Ted, Jung
# Description:
#   Comparison the type of retrievers
#   customed,vector, keyword
# =============================================================================


from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    
    SimpleKeywordTableIndex,
    VectorStoreIndex,
    
    QueryBundle,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import NodeWithScore

from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import List


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
#Settings.llm = Ollama(model="llama3.2", request_timeout=720.0)
Settings.llm = OpenAI(model="gpt-4.1-nano")


documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
parser = SimpleNodeParser() 
nodes = parser.get_nodes_from_documents(documents)


storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)


# Define two indexes(VectorStoreIndex, SimpleKeywordTableIndex) on the Same Data
# VectorStoreIndex for the semantic search
# SimpleKeywordTableIndex for the keyword-based search

vector_index = VectorStoreIndex(nodes)
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)


# Note
# Define Custome Retriever (hybrid search: semantic search and keyword search)
# It returns a set of nodes containing scores
class CustomeRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriver: VectorIndexRetriever,
        keyword_retriver: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:

        """Init params."""

        self._vector_retriever = vector_retriver
        self._keyword_retriever = keyword_retriver

        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        
        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}
        
        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})
        
        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes



# Define Multiple Retrievers
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
custom_retriever = CustomeRetriever(vector_retriever, keyword_retriever)



# Response Mode (Refine, CompactAndRefine,TreeSummarize, etc)
response_synthesizer = get_response_synthesizer()



# Assemble two retrievers in a query engine
# "OR": any node retrieved by either the vector or keyword retriever
# "AND": only includes nodes that are retrieved by both retrivers
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer
)

# vector_retriever(semantic search)
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)

# keyword_retriever(exact keyword matches)
keyword_query_engine = RetrieverQueryEngine(
    retriever=keyword_retriever,
    response_synthesizer=response_synthesizer,
)


response1 = custom_query_engine.query(
    "What did the author do during his time at YC?"
)
print(response1, end="\n\n")

response2 = vector_query_engine.query(
    "What did the author do during his time at YC?"
)
print(response2, end="\n\n")

response3 = keyword_query_engine.query(
    "What did the author do during his time at YC?"
)
print(response3, end="\n\n")
