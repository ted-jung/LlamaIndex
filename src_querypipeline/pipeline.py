# ===========================================================================
# pipeline (Prompt + LLM)
# Created: 19, Feb 2025
# Updated: 17, Mar 2025
# Writer: Ted, Jung
# Description: 
#   QueryPipeline with combination of tool[llm, prompttemplate, output_parser, retriever]
# ===========================================================================

# setup Arize Phoenix for logging/observability
#import phoenix as px
import os

from typing import List
from pydantic import BaseModel, Field

from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.output_parsers import PydanticOutputParser

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)
from llama_index.core import set_global_handler as sgh

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
#Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
# llm = Ollama(model="llama3.2", request_timeout=360.0)
llm = OpenAI(model="gpt-4o-mini", request_timeout=360.0)


#px.launch_app()
#sgh("arize_phoenix")

current_dir = os.getcwd()
reader = SimpleDirectoryReader(f"{current_dir}/data/paul_graham")
docs = reader.load_data()


if not os.path.exists(f"{current_dir}/src_querypipeline/storage"):
    index = VectorStoreIndex.from_documents(docs)
    # save index to disk
    index.set_index_id("vector_index")
    index.storage_context.persist(f"{current_dir}/src_querypipeline/storage")
else:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=f"{current_dir}/src_querypipeline/storage")
    # load index
    index = load_index_from_storage(storage_context, index_id="vector_index")
    
    
# try chaining basic prompts
prompt_str = "Please generate related movies to {movie_name}"
prompt_tmpl = PromptTemplate(prompt_str)

qp = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)

output = qp.run(movie_name="The Departed")
print(str(output))


# output, intermediates = p.run_with_intermediates(movie_name="The Departed")



# This is a structured Pydantic objects for parsing the outputs
class Movie(BaseModel):
    """Object representing a single movie."""

    name: str = Field(..., description="Name of the movie.")
    year: int = Field(..., description="Year of the movie.")


class Movies(BaseModel):
    """Object representing a list of movies."""

    movies: List[Movie] = Field(..., description="List of movies.")



output_parser = PydanticOutputParser(Movies)
json_prompt_str = """\
Please generate related movies to {movie_name}. Output with the following JSON format: 
"""
json_prompt_str = output_parser.format(json_prompt_str)


# add JSON spec to prompt template
json_prompt_tmpl = PromptTemplate(json_prompt_str)


qp = QueryPipeline(chain=[json_prompt_tmpl, llm, output_parser], verbose=True)
output = qp.run(movie_name="Toy Story")

print(str(output))



prompt_str = "Please generate related movies to {movie_name}"
prompt_tmpl = PromptTemplate(prompt_str)
# let's add some subsequent prompts for fun
prompt_str2 = """\
Here's some text:

{text}

Can you rewrite this with a summary of each movie?
"""
prompt_tmpl2 = PromptTemplate(prompt_str2)
llm_c = llm.as_query_component(streaming=True)

qp = QueryPipeline(
    chain=[prompt_tmpl, llm_c, prompt_tmpl2, llm_c], verbose=True
)
# p = QueryPipeline(chain=[prompt_tmpl, llm_c], verbose=True)

output = qp.run(movie_name="The Dark Knight")
for o in output:
    print(o.delta, end="")
    

qp = QueryPipeline(
    chain=[
        json_prompt_tmpl,
        llm.as_query_component(streaming=True),
        output_parser,
    ],
    verbose=True,
)
output = qp.run(movie_name="Toy Story")
print(output)
