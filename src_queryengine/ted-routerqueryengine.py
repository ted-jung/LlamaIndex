# =============================================================================
# RouterQueryEngine
# Created: 22, Jan 2025
# Updated: 11, Mar 2025
# Writer: Ted, Jung
# Description: Booking agent(FunctionCallingAgent) with functiontool
# =============================================================================


from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    SummaryIndex,
    SimpleKeywordTableIndex,
)

from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import (
    LLMMultiSelector,
    LLMSingleSelector,
    PydanticMultiSelector
)


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
Settings.chunk_size = 1024


# load documents
# initialize settings (set chunk size)
documents = SimpleDirectoryReader("./data/").load_data()
nodes = Settings.node_parser.get_nodes_from_documents(documents)


# initialize storage context (by default, in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)


# ceate two query engines
summary_index = SummaryIndex(nodes, storage_context=storage_context)
list_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
vector_query_engine = vector_index.as_query_engine()



# define query engine tools like below for different types of queries
# one for summarization and one for reteriving semantically similar context
list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description=(
        "Useful for summarization questions related to Paul Graham eassy on"
        " What I Worked On."
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)



# Create q query engine(routerqueryengine) with a single selector
# also, having a list of queryengine tools
# test he query engine with diffrent selectors (single, multi, pydantic)
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)
response = query_engine.query("What is the summary of the document?")
print(str(response))


query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)
response = query_engine.query("What is the summary of the document?")
print(str(response))


# add one more index for keyword search
# it uses just a simple regex extractor to extract keywords from the text
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context using keywords from Paul"
        " Graham essay on What I Worked On."
    ),
)


query_engine = RouterQueryEngine(
    selector=PydanticMultiSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
        keyword_tool,
    ],
)

# This query could use either a keyword or vector qury engine, so it will combine responses from bothe
response = query_engine.query(
    "What were noteable events and people from the authors time at Interleaf"
    " and YC?"
)
print(str(response))


# [optional] look at selected results
print(str(response.metadata["selector_result"]))