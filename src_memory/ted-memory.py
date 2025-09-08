# =============================================================================
# Memory Module
# Created: 8, Sep 2025
# Updated: 8, Sep 2025
# Writer: Ted, Jung
# Description: 
#   Memory: Short-term and Long-term memory
#   - Short-term memory: stores recent interactions within the context window of the LLM
#   - Long-term memory: stores important information that can be retrieved later
#
#   Memory blocks: Different types of memory blocks for different use cases
#     1. StaticMemoryBlock: stores static information that does not change
#     2. FactExtractionMemoryBlock: extracts and stores facts from interactions
#     3. VectorMemoryBlock: stores information in a vector store for similarity search
#        - Vector stores: Different vector stores can be used for long-term memory
#           1. ClickHouse
#           2. Chroma
#           3. etc
# =============================================================================


import asyncio
import clickhouse_connect


from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.clickhouse import ClickHouseVectorStore

from llama_index.core.settings import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import (
    Memory,
    StaticMemoryBlock,
    FactExtractionMemoryBlock,
    VectorMemoryBlock,
)

from llama_index.core.agent.workflow import FunctionAgent



Settings.llm = OpenAI(model="gpt-4.1-nano")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# token_limit: The maximum number of tokens to store in memory.
# token_flush_size: The number of tokens to flush to long-term memory when the token limit is exceeded.
# chat_history_token_ratio:  short-term(token_limt*chtr), long-term (token_limt*(1-chtr))
memory = Memory.from_defaults(
    session_id="my_session",
    token_limit=50,  # Normally you would set this to be closer to the LLM context window (i.e. 75,000, etc.)
    token_flush_size=10,
    chat_history_token_ratio=0.7,
)



# Simulate a conversation for short-term ===========

async def ted():
    for i in range(100):
        await memory.aput_messages(
            [
                ChatMessage(role="user", content="Hello, world!"),
                ChatMessage(role="assistant", content="Hello, world to you too!"),
                ChatMessage(role="user", content="What is the capital of France?"),
                ChatMessage(
                    role="assistant", content="The capital of France is Paris."
                ),
            ]
        )


    current_chat_history = await memory.aget()
    for msg in current_chat_history:
        print(msg)


    all_messages = await memory.aget()
    for msg in current_chat_history:
        print(msg)

    all_messages = await memory.aget_all()
    print(len(all_messages))  # Should be 400 messages total

    # memory reset to clear all messages
    await memory.areset()

    all_messages = await memory.aget_all()
    print(len(all_messages))  # Should be 0 messages total



# long-term memory ====================================

# Create a clickhouse client to connect to ClickHouse
async def ted2():
    ch_client = clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        username="default",
        password="magic",
        database="default",
    )

    # Create an empty table in ClickHouse
    vector_store = ClickHouseVectorStore(
        ch_client, 
        table="ted_memory",
        embed_model=Settings.embed_model
    )

    blocks = [
        StaticMemoryBlock(
            name="core_info",
            static_content="My name is Logan, and I live in Saskatoon. I work at LlamaIndex.",
            priority=0,
        ),
        FactExtractionMemoryBlock(
            name="extracted_info",
            llm=Settings.llm,
            max_facts=50,
            priority=1,
        ),
        VectorMemoryBlock(
            name="vector_memory",
            # required: pass in a vector store like qdrant, chroma, weaviate, milvus, etc.
            vector_store=vector_store,
            priority=2,
            embed_model=Settings.embed_model,
            # The top-k message batches to retrieve
            # similarity_top_k=2,
            # optional: How many previous messages to include in the retrieval query
            # retrieval_context_window=5
            # optional: pass optional node-postprocessors for things like similarity threshold, etc.
            # node_postprocessors=[...],
        ),
    ]

    ted_memory = Memory.from_defaults(
        session_id="my_session",
        token_limit=30000,
        # Setting a extremely low ratio so that more tokens are flushed to long-term memory
        chat_history_token_ratio=0.02,
        token_flush_size=500,
        memory_blocks=blocks,
        # insert into the latest user message, can also be "system"
        insert_method="user",
    )




    # If the agent does not have any tools, it uses just llm.
    agent = FunctionAgent(
        tools=[],  # No tools provided
        llm=Settings.llm,
    )

    user_msgs = [
        "Hi! My name is Logan",
        "What is your opinion on minature shnauzers?",
        "Do they shed a lot?",
        "What breeds are comparable in size?",
        "What is your favorite breed?",
        "Would you recommend owning a dog?",
        "What should I buy to prepare for owning a dog?",
    ]

    for user_msg in user_msgs:
        # 1. Add the new user message to the memory
        ted_memory.put(ChatMessage(role=MessageRole.USER, content=user_msg))

        # 2. Run the agent with both the new user message and the memory
        # The agent will use the 'memory' object to access the full chat history.
        _ = await agent.run(user_msg=user_msg  , memory=ted_memory)
        

    chat_history = await ted_memory.aget()
    print(len(chat_history))  # Should be <= token_limit

    for msg in chat_history:
        print(msg)


if __name__ == "__main__":
    asyncio.run(ted())
    asyncio.run(ted2())