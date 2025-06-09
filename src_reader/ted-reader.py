from llama_index.core import VectorStoreIndex, download_loader
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

llm = OpenAI(model="gpt-4.1-nano")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=["https://clickhouse.com/company/careers"])
index = VectorStoreIndex.from_documents(documents)
ch_engine = index.as_query_engine(llm=llm)
response = ch_engine.query("What job openings for united states?")
print(response)