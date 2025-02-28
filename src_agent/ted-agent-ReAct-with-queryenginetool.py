# ===========================================================================
# ReAct Agent
# Date: 27, Feb 2025
# Updated: 27, Feb 2025
# Writer: Ted, Jung
# Description: 1. ReAct(loop) agent for financial analysis
#              2. Tools[a, b, c] <- accessing by agant
#                 one tool - query a, the other tool - query b
#              Why do we use QueryEngineTool? for RAG
# ===========================================================================



import os


from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = OpenAI(model="gpt-4o-mini")
curr_dir = os.getcwd()



# Create Index, set Dir to persist
try:
    storage_context = StorageContext.from_defaults(
        persist_dir=f"{curr_dir}/src_agent/storage/lyft"
    )
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir=f"{curr_dir}/src_agent/storage/uber"
    )
    uber_index = load_index_from_storage(storage_context)

    index_loaded = True
except Exception:
    index_loaded = False



if not index_loaded:
    # load data
    lyft_docs = SimpleDirectoryReader(
        input_files=[f"{curr_dir}/data/pdf/lyft_2021.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=[f"{curr_dir}/data/pdf/uber_2021.pdf"]
    ).load_data()

    # build index
    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    uber_index = VectorStoreIndex.from_documents(uber_docs)

    # persist index
    lyft_index.storage_context.persist(persist_dir=f"{curr_dir}/src_agent/storage/lyft")
    uber_index.storage_context.persist(persist_dir=f"{curr_dir}/src_agent/storage/uber")



lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)


query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]



agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    # context=context  (system prompt)
)

prompt_dict = agent.get_prompts()
for k, v in prompt_dict.items():
    print(f"Prompt: {k} \n\nValue: {v.template}")


response = agent.chat("What was Lyft's revenue growth in 2021?")
print(str(response))

response = agent.chat(
    "Compare and contrast the revenue growth of Uber and Lyft in 2021, then"
    " give an analysis"
)
print(str(response))