import logging
import sys


from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from IPython.display import Markdown, display


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = OpenAI(model="gpt-4.1-nano", request_timeout=720.0)
#Settings.llm = Ollama(model="llama3.2", request_timeout=720.0)



# initialize settings (set chunk size)
Settings.chunk_size = 1024


# load documents and indexing it
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents)



# turn index into queryengine, do a query with question
query_str = "what did paul graham do after going to RISD"
query_engine = index.as_query_engine()


# just a normal query on the query engine
response = query_engine.query(query_str)
print(f"\nnormal vectorstoreindex engine****************")
print(response)



# HyDEQueryTransform works by generating a hypothetical document based on the query
# and using it for embedding lookup
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)
response = hyde_query_engine.query(query_str)
print(f"\nhypotetical document for embedding****************")
print(response)


query_bundle = hyde(query_str)
hyde_doc = query_bundle.embedding_strs[0]

print(f"\nthis is the hyde docs****************")
print(hyde_doc)
