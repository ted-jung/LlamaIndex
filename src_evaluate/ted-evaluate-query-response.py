# ===========================================================================
# Evaluation (faithfulness, relevancy)
# Date: 30, Jan 2025
# Updated: 27, Feb 2025
# Writer: Ted, Jung
# Description: 1. Generate question on source file(essay) by LLM1
#              2. Evaluate F,R in Batch
# ===========================================================================


import os
import asyncio

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, load_index_from_storage
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, BatchEvalRunner

from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.storage import StorageContext

# Define llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", temperature=0.0, request_timeout=720.0)
gpt_4o_mini = OpenAI(temperature=0, model="gpt-4o-mini")


# Define evaluator for (faithfulness, relevancy)
faithfulness_evaluator = FaithfulnessEvaluator(llm=gpt_4o_mini)
relevancy_evaluator = RelevancyEvaluator(llm=gpt_4o_mini)



# Create an index from documents
curr_dir = os.getcwd()
persist_dir = f"{curr_dir}/src_evaluate/persist/"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
documents = SimpleDirectoryReader(input_files=["./data/paul_graham/paul_graham_essay_short.txt"]).load_data()

if os.path.exists(persist_dir):
    vector_index = load_index_from_storage(storage_context, llm = llm)
else:    
    vector_index = VectorStoreIndex.from_documents(documents=documents)
    vector_index.storage_context.persist(persist_dir=persist_dir)


# Define generator & generate questions using documents
dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    llm=gpt_4o_mini,
    num_questions_per_chunk=2,  # set the number of questions per nodes
)


# Question generate from source essay.txt by LLM
rag_dataset = dataset_generator.generate_questions_from_nodes()
questions = [e.query for e in rag_dataset.examples]


# Use BatchEvalRunner
runner = BatchEvalRunner(
    {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
    workers=8,
)


async def ted():
    eval_results = await runner.aevaluate_queries(
        vector_index.as_query_engine(), queries=questions
    )

    for key in eval_results['faithfulness']:
        for i in key:
            print(f"Key: {i}")



if __name__ == "__main__":
    asyncio.run(ted())