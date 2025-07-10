# =============================================================================
# Title: Multi-LLM 
# Created: 10, July 2025
# Updated: 10, July 2025
# Writer: Ted, Jung
# Decscription: Multi-LLM from Anthropic (use cases)
#       - Just read an image to describe it
#       - Use MultiModal(Anthropic) to reason images from URLs
#       - Structured output parsing from an image
# =============================================================================


import os
import matplotlib.pyplot as plt
import requests
import matplotlib.pyplot as plt
import qdrant_client


from PIL import Image
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    StorageContext,
)
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.schema import TextNode


from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal


from io import BytesIO
from pydantic import BaseModel
from typing import List
from pathlib import Path


from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.qdrant import QdrantVectorStore



os.environ["ANTHROPIC_API_KEY"] = ""  # Your ANTHROPIC API key here
img = Image.open("../data/images/prometheus_paper_card.png")
plt.imshow(img)



# put your local directory here
# SimpleDirectoryReader can read multiple files in a directory
# like this: image, pdf, text, md, csv, json, etc.
# For this example, we will use a single image file.
image_documents = SimpleDirectoryReader(
    input_files=["../data/images/prometheus_paper_card.png"]
).load_data()


# Initiated Anthropic MultiModal class
anthropic_mm_llm = AnthropicMultiModal(max_tokens=300)

response = anthropic_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

print(response)



# Use AnthropicMultiModal to reason images from URLs

image_urls = [
    "https://venturebeat.com/wp-content/uploads/2024/03/Screenshot-2024-03-04-at-12.49.41%E2%80%AFAM.png",
    # Add yours here!
]

response = requests.get(image_urls[0])
print(f"response status code: {response.status_code}")
img = Image.open(BytesIO(response.content))
plt.imshow(img)


# Load image and convert to image documents
# Question about iamge content
image_url_documents = load_image_urls(image_urls)


response = anthropic_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_url_documents,
)

print(response)




# put your local directore here
image_documents = SimpleDirectoryReader(
    input_files=["../data/images/ark_email_sample.PNG"]
).load_data()

img = Image.open("../data/images/ark_email_sample.PNG")
plt.imshow(img)



# Declare two pydantic classes for strcutured output parsing
# In general, multiple objects in a single object
class TickerInfo(BaseModel):
    """List of ticker info."""

    direction: str
    ticker: str
    company: str
    shares_traded: int
    percent_of_total_etf: float


class TickerList(BaseModel):
    """List of stock tickers."""

    fund: str
    tickers: List[TickerInfo]




prompt_template_str = """\
Can you get the stock information in the image \
and return the answer? Pick just one fund. 

Make sure the answer is a JSON format corresponding to a Pydantic schema. The Pydantic schema is given below.

"""

# Initiated Anthropic MultiModal class
anthropic_mm_llm = AnthropicMultiModal(max_tokens=300)



# Different "Program" Types: Within llama_index.core.program, 
# you'll find different types of "programs" (like LLMTextCompletionProgram, 
# FunctionCallingProgram, etc.) that cater to various ways of interacting with LLMs for structured output. 
# Some use traditional text completion and then parse the output, 
# while others leverage advanced LLM capabilities like function calling to directly generate structured data.

llm_program = MultiModalLLMCompletionProgram.from_defaults(
    output_cls=TickerList,
    image_documents=image_documents,
    prompt_template_str=prompt_template_str,
    multi_modal_llm=anthropic_mm_llm,
    verbose=True,
)

response = llm_program()


print(str(response))



# Index into a vector store

anthropic_mm_llm = AnthropicMultiModal(max_tokens=300)



# Load images to read the images and create text
# turn text into nodes and put it into a list
nodes = []
for img_file in Path("mixed_wiki_images_small").glob("*.png"):
    print(img_file)
    
    image_documents = SimpleDirectoryReader(input_files=[img_file]).load_data()
    response = anthropic_mm_llm.complete(
        prompt="Describe the images as an alternative text",
        image_documents=image_documents,
    )
    metadata = {"img_file": img_file}
    nodes.append(TextNode(text=str(response), metadata=metadata))


# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_mixed_img")
vector_store = QdrantVectorStore(client=client, collection_name="collection")

# Using the embedding model to Gemini
embed_model = OpenAIEmbedding()
anthropic_mm_llm = AnthropicMultiModal(max_tokens=300)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)


# now you can query the index where the converted text from images is stored
# and use Antropic to query the index
query_engine = index.as_query_engine(llm=Anthropic())
response = query_engine.query("Tell me more about the porsche")
print(str(response))