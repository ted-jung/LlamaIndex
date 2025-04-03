# ===========================================================================
# LlamaParse - how to use?
# Created: 4, Mar 2025
# Updated: 4, Mar 2025
# Writer: Ted, Jung
# Description: 
#   LlamaParse
# ===========================================================================



# from llama_cloud_services import LlamaParse

import os

from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

load_dotenv()

curr_dir = os.getcwd()

# print(os.environ.get("LLAMA_CLOUD_API_KEY"))
# documents = LlamaParse(result_type="markdown").load_data(
#     f"{curr_dir}/data/pdf/uber_10q_march_2022.pdf"
# )
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader(input_files=[f"{curr_dir}/data/pdf/2023_canadian_budget.pdf"], file_extractor=file_extractor).load_data()

print(documents)