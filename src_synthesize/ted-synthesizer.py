# =============================================================================
# Title: Synthesizer
# Created: 30, Apr 2025
# Updated: 30, Apr 2025
# Writer: Ted Jung
# Description: 
#       How to use synthesizer(treesummarize, refine) with structured outputs. 
#       Specifically, 
#       TreeSummarize is used to output pydantic objects
#       : prompt -> synthesizer(TS) -> answer
#       Refine
#       : prompt -> get answer -> prompt(refined) -> get answer
# =============================================================================


import os
import openai


from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import TreeSummarize, Refine
from llama_index.core.types import BaseModel
from typing import List


llm = OpenAI(model="gpt-4o-mini")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


reader = SimpleDirectoryReader(input_files=["./data/paul_graham/paul_graham_essay.txt"])
docs = reader.load_data()
text = docs[0].text


# index = VectorStoreIndex.from_documents(documents=docs)
# print(text)


# Define a PromptTemplate (context, tone, query)
qa_prompt_tmpl = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Please also write the answer in the style of {tone_name}.\n"
    "Query: {query_str}\n"
    "Answer: "
)

qa_prompt = PromptTemplate(qa_prompt_tmpl)


# a Refined prompt
refine_prompt_tmpl = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "Please also write the answer in the style of {tone_name}.\n"
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)
refine_prompt = PromptTemplate(refine_prompt_tmpl)



# Two Synthesizers
# 1. TreeSummarize: response builder
#    : Repack the chunk from leaves to root (bottom-up approach)
# 2. Refine: refine response
#    : First prompt to get the initial node and do iteratively with consecutine nodes
#      to refine it to finalize answer

summarizer = TreeSummarize(verbose=True, summary_template=qa_prompt)

response = summarizer.get_response(
    query_str="who is Paul Graham?", 
    text_chunks=[text], 
    tone_name="a Shakespeare play"
)


print(str(response))

summarizer = Refine(
    verbose=True,
    text_qa_template=qa_prompt,
    refine_template=refine_prompt
)

response = summarizer.get_response(
    "who is Paul Graham?", [text], tone_name="a haiku"
)

print(str(response))


# Try with PyDantic Model
# Define a new pydantic class

class Biography(BaseModel):
    """Data model for a biography"""

    name: str
    best_known_for: List[str]
    extra_info: str
    age: int


# Create pydantic model to structure response

summarizer = TreeSummarize(
    verbose = True,
    summary_template=qa_prompt,
    output_cls=Biography
)

response = summarizer.get_response(
    query_str="who is Paul Graham?", 
    text_chunks=[text], 
    tone_name="a business memo"
)

print(str(response))