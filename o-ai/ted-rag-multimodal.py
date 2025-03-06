# ===========================================================================
# RAG - MultiModal
# Created: 28, Feb 2025
# Updated: 4, Mars 2025
# Writer: Ted, Jung
# Description: 
#   1. MultiModal index & LLM for Multimodal
#   2. notice: CH (text-768, image-512)
# ===========================================================================



import os
import time
from tkinter import image_types
import requests
import urllib.request
import matplotlib.pyplot as plt
import clickhouse_connect

from pathlib import Path
from PIL import Image

from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.indices import MultiModalVectorStoreIndex

from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import SimpleMultiModalQueryEngine

from llama_index.core.response.notebook_utils import display_source_node

# from transformers import CLIPConfig
# config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
# text_hidden_size = config.text_config.hidden_size
# vision_hidden_size = config.vision_config.hidden_size



curr_dir = os.getcwd()


# Rome colosseum at night
image_urls = [
    "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",
]
image_documents = load_image_urls(image_urls)


embed_model_txt = HuggingFaceEmbedding(model_name="openai/clip-vit-base-patch32")
embed_model_img = HuggingFaceEmbedding(model_name="openai/clip-vit-base-patch32")
# embed_model_txt = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# embed_model_img = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.embed_model = embed_model_txt

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4o-mini", max_new_tokens=300
)


response = openai_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)
print(response)



input_image_path = Path("input_images")
if not input_image_path.exists():
    Path.mkdir(input_image_path)



def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 5:
                break

def plot_images2(image_meta):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path, img_type in image_meta:
        if os.path.isfile(img_path) and img_type =="image/jpeg":
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            images_shown += 1
            if images_shown >= 5:
                break



image_paths = []
for img_path in os.listdir(f"{curr_dir}/o-ai/input_images"):
    image_paths.append(str(os.path.join(f"{curr_dir}/o-ai/input_images", img_path)))

plot_images(image_paths)





# put your local directore here
image_documents = SimpleDirectoryReader(f"{curr_dir}/o-ai/input_images").load_data()

response = openai_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)
print(response)



def get_wikipedia_images(title):
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|dimensions|mime",
            "generator": "images",
            "gimlimit": "50",
        },
    ).json()
    image_urls = []
    for page in response["query"]["pages"].values():
        if page["imageinfo"][0]["url"].endswith(".jpg") or page["imageinfo"][
            0
        ]["url"].endswith(".png"):
            image_urls.append(page["imageinfo"][0]["url"])
    return image_urls




# Download text and images from Wikipedia

image_uuid = 0
# image_metadata_dict stores images metadata including image uuid, filename and path
image_metadata_dict = {}
MAX_IMAGES_PER_WIKI = 5

wiki_titles = {
    "Tesla Model Y",
    "Tesla Model X",
    # "Tesla Model 3",
    # "Tesla Model S",
    "Kia EV6",
    # "BMW i3",
    # "Audi e-tron",
    # "Ford Mustang",
    "Porsche Taycan",
    # "Rivian",
    # "Polestar",
}

data_path = Path(f"{curr_dir}/o-ai/mixed_wiki")
if not data_path.exists():
    Path.mkdir(data_path)

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

    images_per_wiki = 0
    try:
        # page_py = wikipedia.page(title)
        list_img_urls = get_wikipedia_images(title)

        # print(list_img_urls)

        for url in list_img_urls:
            if (
                url.endswith(".jpg")
                or url.endswith(".png")
                or url.endswith(".svg")
            ):
                image_uuid += 1
                # image_file_name = title + "_" + url.split("/")[-1]

                # download and save it
                urllib.request.urlretrieve(
                    url, data_path / f"{image_uuid}.jpg"
                )
                time.sleep(1)
                images_per_wiki += 1
                # Limit the number of images downloaded per wiki page to 15
                if images_per_wiki > MAX_IMAGES_PER_WIKI:
                    break
    except Exception as e:
        print(e)
        print(
            "Number of images found for Wikipedia page: {} are {}".format(
                title, images_per_wiki
            )
        )
        continue



# Create the MultiModal index

# ClickHouse Vector Store
ch_client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    username="default",
    password="",
    database="default",
)


# Prepare the target table engine on CH
text_store = ClickHouseVectorStore(
    ch_client, 
    table="text_collection",
    embed_model=embed_model_txt
)
image_store = ClickHouseVectorStore(
    ch_client, 
    table="image_collection",
    embed_model=embed_model_img
)


# StorageContext having two vector_stores(vector_store=vector_store)
# Read data & persist it
storage_context = StorageContext.from_defaults(
    vector_store=text_store, 
    image_store=image_store
)
documents = SimpleDirectoryReader(f"{curr_dir}/o-ai/mixed_wiki/").load_data()
storage_context.persist()


# Create MM index using two vectorstores
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)



qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, respond with both text and relevant images.\n"
    "If there are images, display them. If there are no images, just return text.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

query_engine = index.as_query_engine(
    llm=openai_mm_llm, text_qa_template=qa_tmpl
)
query_str = "Tell me more about the Porsche Taycan"
response = query_engine.query(query_str)
print(f"Response: {response.response}") # print the whole response.

for node in response.source_nodes:
    print(f"Node Metadata: {node.metadata}")


def ted_display_source_node(text_node, source_length=200):
    """Prints the source node information."""
    print("Source Node:")
    print(f"  Node ID: {text_node.node_id}")
    print(f"  Text: {text_node.get_content()[:source_length]}...")  # Display first 200 chars
    print(f"**Similarity:** {text_node.score}<br>")
    print(f"  Metadata: {text_node.metadata}")
    # Add more details as needed



for text_node in response.metadata["text_nodes"]:
    ted_display_source_node(text_node, source_length=200)


plot_images2(
    [(n.metadata["file_path"], n.metadata["file_type"])  for n in response.metadata["text_nodes"]]
)
