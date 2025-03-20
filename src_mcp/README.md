# MCP

MCP(Model Context Protocol) is an open protocol that standardizes how applications provide context to LLMs.

MCP provides a standardized way to connect AI models to different data sources and tools.

MCP is indeed a methodology for structuring interactions with Large Language Models (LLMs) by systematically managing context. 

It defines how prompts, memory, retrieval, and updates should be handled to ensure coherent, efficient, and structured interactions with LLMs.

## Key Aspects of MCP
MCP provides a framework for:

Defining Context Windows – Ensuring relevant history and knowledge are included in prompts.

Retrieval & Augmentation – Efficiently pulling in data from external sources like vector databases (e.g., LlamaIndex, LangChain).

Memory Management – Deciding what past interactions should persist and how they should be formatted.

Adaptive Prompting – Structuring interactions dynamically based on user intent and prior conversation history.

Multi-Agent Interactions – Coordinating between multiple LLMs or sub-models.

## How MCP Works in LLM Interactions

Prompt Engineering: Formatting the input prompt to include the right mix of user queries, past context, and retrieved knowledge.

Memory Handling: Deciding what historical interactions or knowledge should persist across sessions.

Knowledge Injection: Using RAG (Retrieval-Augmented Generation) to fetch relevant context from databases, wikis, or documents.

Execution Protocol: Structuring how LLMs interact with APIs, databases, and users dynamically.


## MCP in Action

Suppose you're designing a chatbot that answers business questions.

Using MCP, you would:

Store past queries and answers for continuity.
Retrieve relevant documents before querying the LLM.
Format the prompt optimally 
(e.g., “User asked about revenue trends. Retrieve sales data and summarize insights.”).

Decide when to refresh context to avoid token overflow.


## MCP & LlamaIndex 🚀

If you're working with LlamaIndex, MCP can be implemented by:

1. Using memory modules to retain long-term context.

2. Employing query transformations to dynamically adjust prompts.

3. Structuring retrieval logic to fetch the right context before an LLM call.



# MCP Implementation Overview

## Example1 - datasource

1. Store and manage context with a persistent index.

2. Retrieve relevant data dynamically before querying the LLM.

3. Use memory modules to maintain conversation continuity.

4. Structure the interaction pipeline using LlamaIndex.


### Install dependency

```
    > pip install llama-index llama-index-llms-openai openai
```


### Define Context & Memory

```
    from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
    from llama_index.llms import OpenAI
    from llama_index.memory import ChatMemoryBuffer

    # Load Documents (Simulating Long-Term Knowledge)
    documents = SimpleDirectoryReader("data/").load_data()

    # Initialize LLM
    llm = OpenAI(model="gpt-4o-mini")

    # Create Service Context
    service_context = ServiceContext.from_defaults(llm=llm)

    # Create Index & Memory
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    retriever = index.as_retriever()
    memory = ChatMemoryBuffer.from_defaults()

```

### MCP Query Pipeline

✅ Retrieve past conversation memory

✅ Fetch relevant knowledge from index

✅ Format the prompt dynamically

✅ Query LLM with optimized context

```
    def mcp_query(user_input):
        """Implements Model Context Protocol (MCP) by managing context dynamically."""
        
        # Retrieve memory context
        past_messages = memory.get()
        memory_context = "\n".join([f"User: {msg.content}" for msg in past_messages[-5:]])  # Last 5 interactions
        
        # Retrieve external knowledge
        retrieved_docs = retriever.retrieve(user_input)
        retrieved_text = "\n".join([doc.text[:500] for doc in retrieved_docs])  # First 500 chars per doc
        
        # Format the prompt with context
        prompt = f"""
        You are a business AI assistant following the Model Context Protocol (MCP).
        
        Conversation History:
        {memory_context}
        
        Retrieved Knowledge:
        {retrieved_text}
        
        User Query:
        {user_input}
        
        Provide a precise and well-structured response based on context.
        """
        
        # Query LLM
        response = llm.complete(prompt)
        
        # Update memory
        memory.put(user_input, response.text)
        
        return response.text

```

### Test MCP Workflow

```
    user_input = "What are the latest hiring trends in AI?"
    response = mcp_query(user_input)
    print(response)
```


## Example2 - using tools

we can dynamically fetch context from external sources (APIs, databases, etc.), making interactions more adaptive and context-aware.

This is how MCP works with Tools

Memory: Stores past conversation history.

Retrieval: Searches relevant documents.

Tool Use: Calls the job market API dynamically when relevant.

LLM Query: Constructs an optimized prompt with structured context.


### MCP with LlamaIndex Tools

✅ Memory – Maintain past interactions

✅ Retrieval – Search relevant documents

✅ Tools – Query external APIs dynamically

✅ LLM Query – Generate responses with structured context

#### Step1: Install Dependency

```
    > pip install llama-index llama-index-llms-openai openai
```

#### Step2: Define Context & Tools

will create a custom tool that fetch job market data from API

```
    from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
    from llama_index.llms import OpenAI
    from llama_index.memory import ChatMemoryBuffer
    from llama_index.tools import FunctionTool
    import requests

    # Load documents (Simulating knowledge base)
    documents = SimpleDirectoryReader("data/").load_data()
    index = VectorStoreIndex.from_documents(documents)

    # Setup OpenAI LLM
    llm = OpenAI(model="gpt-4")
    service_context = ServiceContext.from_defaults(llm=llm)

    # Setup memory
    memory = ChatMemoryBuffer.from_defaults()

    # Define a job market analysis tool (Fetching external data)
    def fetch_job_market(country: str):
        """Fetches job market trends for a given country."""
        api_url = f"https://job-market-api.com/{country}"  # Fake API example
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        return {"error": "Failed to fetch data"}

    job_market_tool = FunctionTool.from_defaults(fn=fetch_job_market)
```

#### Step3: MCP Query Pipeline

```
    def mcp_query(user_input):
        """Implements MCP by managing memory, retrieval, and tools dynamically."""
        
        # Retrieve past memory context
        past_messages = memory.get()
        memory_context = "\n".join([f"User: {msg.content}" for msg in past_messages[-3:]])  # Last 3 messages
        
        # Retrieve documents
        retriever = index.as_retriever()
        retrieved_docs = retriever.retrieve(user_input)
        retrieved_text = "\n".join([doc.text[:300] for doc in retrieved_docs])  # First 300 chars per doc
        
        # Check if the user is asking about job trends
        if "job" in user_input.lower() or "hiring" in user_input.lower():
            country = "USA"  # Default country (modify as needed)
            job_data = job_market_tool(country)
            job_market_context = str(job_data)
        else:
            job_market_context = "No external job data needed."
        
        # Format the prompt with structured context
        prompt = f"""
        You are an AI assistant following the Model Context Protocol (MCP).
        
        Conversation History:
        {memory_context}
        
        Retrieved Knowledge:
        {retrieved_text}
        
        External Data (if applicable):
        {job_market_context}
        
        User Query:
        {user_input}
        
        Provide an insightful and well-structured response.
        """
        
        # Query LLM
        response = llm.complete(prompt)
        
        # Update memory
        memory.put(user_input, response.text)
        
        return response.text

```

#### Step4: Test MCP with tools

```
    user_input = "What are the latest job market trends in the USA?"
    response = mcp_query(user_input)
    print(response)
```
