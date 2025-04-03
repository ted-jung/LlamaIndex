# ===========================================================================
# Custom prompt
# Created: 2, Apr 2025
# Updated: 2, Apr 2025
# Writer: Ted, Jung
# Description: response mode on different engine
#              Each engine has a different template
#              can also apply a modified template to get the intended resposne 
# ===========================================================================

import os
import asyncio
import numpy as np

from llama_index.core import (
    Settings, 
    VectorStoreIndex,
    Document
)
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import (
    QueryResponseDataset,
    CorrectnessEvaluator, 
    BatchEvalRunner,
)
from llama_index.core.evaluation.eval_utils import aget_responses
from llama_index.core.prompts import RichPromptTemplate



Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = OpenAI(model="gpt-4o-mini", request_timeout=720.0)
Settings.llm = llm


curr_dir = os.getcwd()


# Read all text from pdf and combine all documents into one
# split the document into chunks of 1024 tokens
docs0 = PyMuPDFReader().load_data(f"{curr_dir}/data/pdf/llama2.pdf")
doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]
node_parser = SentenceSplitter(chunk_size=1024)
base_nodes = node_parser.get_nodes_from_documents(docs)


# Indexing it into VectorStoreIndex and Turn it into a QueryEngine
index = VectorStoreIndex(base_nodes)
query_engine = index.as_query_engine(similarity_top_k=2)


# Golden Dataset
eval_dataset = QueryResponseDataset.from_json(f"{curr_dir}/data/json/llama2_eval_qr_dataset.json")


# Get Evaluator
evaluator_c = CorrectnessEvaluator()
evaluator_dict = {"correctness": evaluator_c}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)



# Define Correctness Eval Function
async def get_correctness(query_engine, eval_qa_pairs, batch_runner):
    # then evaluate
    # TODO: evaluate a sample of generated results
    eval_qs = [q for q, _ in eval_qa_pairs]
    eval_answers = [a for _, a in eval_qa_pairs]
    pred_responses = await aget_responses(
        eval_qs, query_engine, show_progress=True
    )

    eval_results = await batch_runner.aevaluate_responses(
        eval_qs, responses=pred_responses, reference=eval_answers
    )
    avg_correctness = np.array(
        [r.score for r in eval_results["correctness"]]
    ).mean()
    return avg_correctness



emotion_stimuli_dict = {
    "ep01": "Write your answer and give me a confidence score between 0-1 for your answer. ",
    "ep02": "This is very important to my career. ",
    "ep03": "You'd better be sure.",
    # add more from the paper here!!
}

# NOTE: ep06 is the combination of ep01, ep02, ep03
emotion_stimuli_dict["ep06"] = (
    emotion_stimuli_dict["ep01"]
    + emotion_stimuli_dict["ep02"]
    + emotion_stimuli_dict["ep03"]
)



qa_tmpl_str = """\
Context information is below. 
---------------------
{{ context_str }}
---------------------
Given the context information and not prior knowledge, \
answer the query.
{{ emotion_str }}
Query: {{ query_str }}
Answer: \
"""
qa_tmpl = RichPromptTemplate(qa_tmpl_str)


QA_PROMPT_KEY = "response_synthesizer:text_qa_template"


async def run_and_evaluate(
    query_engine, 
    eval_qa_pairs, 
    batch_runner, 
    emotion_stimuli_str, 
    qa_tmpl):
    """Run and evaluate."""
    new_qa_tmpl = qa_tmpl.partial_format(emotion_str=emotion_stimuli_str)

    old_qa_tmpl = query_engine.get_prompts()[QA_PROMPT_KEY]
    query_engine.update_prompts({QA_PROMPT_KEY: new_qa_tmpl})
    avg_correctness = await get_correctness(
        query_engine, eval_qa_pairs, batch_runner
    )
    query_engine.update_prompts({QA_PROMPT_KEY: old_qa_tmpl})
    return avg_correctness


async def ted():
    # try out ep01
    correctness_ep01 = await run_and_evaluate(
        query_engine,
        eval_dataset.qr_pairs,
        batch_runner,
        emotion_stimuli_dict["ep01"],
        qa_tmpl,
    )

    print(correctness_ep01)



    correctness_ep02 = await run_and_evaluate(
        query_engine,
        eval_dataset.qr_pairs,
        batch_runner,
        emotion_stimuli_dict["ep02"],
        qa_tmpl,
    )

    print(correctness_ep02)


    correctness_ep03 = await run_and_evaluate(
        query_engine,
        eval_dataset.qr_pairs,
        batch_runner,
        emotion_stimuli_dict["ep03"],
        qa_tmpl,
    )

    print(correctness_ep03)


asyncio.run(ted())