# ===================================================================
# Agent for planning
# Created: 28, Feb 2025
# Updated: 28, Feb 2025
# Writer: Ted, Jung
# Description: Meaning of Structured planning
#   decompose initial input/task -> several sub-tasks -> ReAct loop
#   -> a set of function calls and thoughts
# ===================================================================



import os
import nest_asyncio

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,Settings
from llama_index.core.tools import QueryEngineTool

from llama_index.core.agent import (
    StructuredPlannerAgent,
    FunctionCallingAgentWorker,
    ReActAgentWorker,
)



# Use ollama in JSON mode
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")



curr_dir = os.getcwd()
# Load documents, create tools
lyft_documents = SimpleDirectoryReader(input_files=[f"{curr_dir}/data/pdf/lyft_2021.pdf"]).load_data()
uber_documents = SimpleDirectoryReader(input_files=[f"{curr_dir}/data/pdf/uber_2021.pdf"]).load_data()

lyft_index = VectorStoreIndex.from_documents(lyft_documents)
uber_index = VectorStoreIndex.from_documents(uber_documents)

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



response = agent.chat(
    "Summarize the key risk factors for Lyft and Uber in their 2021 10-K filings."
)

print(str(response))


DEFAULT_INITIAL_PLAN_PROMPT = """\
Think step-by-step. Given a task and a set of tools, create a comprehesive, end-to-end plan to accomplish the task.
Keep in mind not every task needs to be decomposed into multiple sub-tasks if it is simple enough.
The plan should end with a sub-task that satisfies the overall task.

The tools available are:
{tools_str}

Overall Task: {task}
"""

DEFAULT_PLAN_REFINE_PROMPT = """\
Think step-by-step. Given an overall task, a set of tools, and completed sub-tasks, update (if needed) the remaining sub-tasks so that the overall task can still be completed.
The plan should end with a sub-task that satisfies the overall task.
If the remaining sub-tasks are sufficient, you can skip this step.

The tools available are:
{tools_str}

Overall Task:
{task}

Completed Sub-Tasks + Outputs:
{completed_outputs}

Remaining Sub-Tasks:
{remaining_sub_tasks}
"""


agent = StructuredPlannerAgent(
    worker,
    tools=[lyft_tool, uber_tool],
    initial_plan_prompt=DEFAULT_INITIAL_PLAN_PROMPT,
    plan_refine_prompt=DEFAULT_PLAN_REFINE_PROMPT,
    verbose=True,
)

