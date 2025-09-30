# ===========================================================================
# Evaluation
# Date: 30, Jan 2025
# Updated: 27, Feb 2025
# Writer: Ted, Jung
# Description: 
#    Evaluate the Query Engine
#    evaluates whether a response is faithful(i.g, hallucinated?) to the contexts
#    Two evaluation
#    - Response: Response <-> Retrieved Contexts, Query, Guidelines
#    - Retrieval: Source <-> Query
# ===========================================================================


from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.evaluation import (
    FaithfulnessEvaluator, 
    RelevancyEvaluator, 
    RetrieverEvaluator
)

from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Env set for LLM
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
# llm = Ollama(model="llama3.2", temperature=0.0, request_timeout=720.0)


# Build index
documents = SimpleDirectoryReader(input_files=["./data/paul_graham/paul_graham_essay.txt"]).load_data()
vector_index = VectorStoreIndex.from_documents(documents=documents)


# Define FaithfullnewwEvaluator
evaluator = FaithfulnessEvaluator(llm=llm)


# Query Index
# response(response + source)
query_engine = vector_index.as_query_engine()
response = query_engine.query(
    "What two things did Paul Graham before college?"
)
eval_result = evaluator.evaluate_response(response=response)
print(eval_result)
print(str(eval_result.passing))


# Retriev Index
retriever = vector_index.as_retriever(similarity_top_k=2)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)
eval_result2 = retriever_evaluator.evaluate(
    query="query", expected_ids=["node_id1", "node_id2"]
)
print(eval_result2)
