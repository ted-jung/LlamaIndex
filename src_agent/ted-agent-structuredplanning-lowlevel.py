# ===================================================================
# Agent for planning (lowlevel)
# Created: 28, Feb 2025
# Updated: 28, Feb 2025
# Writer: Ted, Jung
# Description: Meaning of Structured planning
#   decompose initial input/task -> several sub-tasks -> ReAct loop
#   -> a set of function calls and thoughts
#
#   To expose the underlying plan,tasks, etc to human to modify them
# ===================================================================



import os
import asyncio
import nest_asyncio

from llama_index.llms.openai import OpenAI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    Settings, 
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool

from llama_index.core.agent import (
    StructuredPlannerAgent,
    FunctionCallingAgentWorker,
    ReActAgentWorker,
)

from llama_index.core.storage import StorageContext


curr_dir = os.getcwd()

# Use ollama in JSON mode
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")



# Load documents, create tools
try:
    storage_context = StorageContext.from_defaults(persist_dir=f"{curr_dir}/src_agent/storage/lyft/")
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(persist_dir=f"{curr_dir}/src_agent/storage/uber/")
    uber_index = load_index_from_storage(storage_context)

    is_loaded = True
except:
    is_loaded = False 


# if os.path.exists(curr_dir) and os.path.isdir(curr_dir):
if not is_loaded:
    lyft_documents = SimpleDirectoryReader(
        input_files=[f"{curr_dir}/data/pdf/lyft_2021.pdf"]
        ).load_data()
    
    uber_documents = SimpleDirectoryReader(
        input_files=[f"{curr_dir}/data/pdf/uber_2021.pdf"]
        ).load_data()

    lyft_index = load_index_from_storage(storage_context)
    uber_index = load_index_from_storage(storage_context)

    lyft_index = VectorStoreIndex.from_documents(lyft_documents)
    uber_index = VectorStoreIndex.from_documents(uber_documents)

    lyft_index.storage_context.persist(persist_dir=f"{curr_dir}/src_agent/storage/lyft/")
    uber_index.storage_context.persist(persist_dir=f"{curr_dir}/src_agent/storage/uber/")


lyft_tool = QueryEngineTool.from_defaults(
    lyft_index.as_query_engine(),
    name="lyft_2021",
    description="Useful for asking questions about Lyft's 2021 10-K filling.",
)

uber_tool = QueryEngineTool.from_defaults(
    uber_index.as_query_engine(),
    name="uber_2021",
    description="Useful for asking questions about Uber's 2021 10-K filling.",
)



# create the function calling worker for reasoning
worker = FunctionCallingAgentWorker.from_tools(
    [lyft_tool, uber_tool], 
    verbose=True
)

# wrap the worker in the top-level planner
agent = StructuredPlannerAgent(
    worker, 
    tools=[lyft_tool, uber_tool], 
    verbose=True
)



nest_asyncio.apply()



plan_id = agent.create_plan(
    "Summarize the key risk factors for Lyft and Uber in their 2021 10-K filings."
)

plan = agent.state.plan_dict[plan_id]

for sub_task in plan.sub_tasks:
    print(f"===== Sub Task {sub_task.name} =====")
    print("Expected output: ", sub_task.expected_output)
    print("Dependencies: ", sub_task.dependencies)



# Execute the first set of tasks
next_tasks = agent.state.get_next_sub_tasks(plan_id)

for sub_task in next_tasks:
    print(f"===== Sub Task {sub_task.name} =====")
    print("Expected output: ", sub_task.expected_output)
    print("Dependencies: ", sub_task.dependencies)


for sub_task in next_tasks:
    response = agent.run_task(sub_task.name)
    agent.mark_task_complete(plan_id, sub_task.name)



# Step-wise execution per task

for sub_task in next_tasks:
    # get the task from the state 
    task = agent.state.get_task(sub_task.name)

    # run intial resoning step
    step_output = agent.run_step(task.task_id)

    # loop until the last step is reached
    while not step_output.is_last:
        step_output = agent.run_step(task.task_id)
    
    # finalize the response and commit to memory
    agent.finalize_response(task.task_id, step_output=step_output)



# check if we are done
next_tasks = agent.get_next_tasks(plan_id)
print(len(next_tasks))


for sub_task in next_tasks:
    print(f"===== Sub Task {sub_task} =====")



# Refine the plan
agent.refine_plan(
    "Summarize the key risk factors for Lyft and Uber in their 2021 10-K filings.",
    plan_id,``
)


plan = agent.state.plan_dict[plan_id]

for sub_task in plan.sub_tasks:
    print(f"===== Sub Task {sub_task.name} =====")
    print("Expected output: ", sub_task.expected_output)
    print("Dependencies: ", sub_task.dependencies)


# Loop until done
async def ted():
    while True:
        # are we done?
        next_tasks = agent.get_next_tasks(plan_id)
        if len(next_tasks) == 0:
            break

        # run concurrently for better performance
        responses = await asyncio.gather(
            *[agent.arun_task(task_id) for task_id in next_tasks]
        )
        for task_id in next_tasks:
            agent.mark_task_complete(plan_id, task_id)

        # refine the plan
        await agent.arefine_plan(
            "Summarize the key risk factors for Lyft and Uber in their 2021 10-K filings.",
            plan_id,
        )


    print(str(responses[-1]))



if __name__ == "__main__":
    asyncio.run(ted())