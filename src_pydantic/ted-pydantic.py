import json
import os

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer

from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


from llama_index.embeddings.huggingface import HuggingFaceEmbedding


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


# --- Configuration ---
# Set your OpenAI API key here.
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


# Get sources from the web using headless browser
def ted_source_data(url)-> str:

    job_country_string = ""
    options = Options()
    options.add_argument("--headless")    # Run Chrome in headless mode
    options.add_argument("--disable-gpu") # Recommended for headless

    driver = webdriver.Chrome(options=options)
    driver.get(url)


    # Wait for the data to load (important!)
    # Example: Wait for an element to appear (replace with your selector, mb-16, mb-1, text-ne~~~)
    # CSS? replace with your web site
    try:
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, ".mb-16")) 
        WebDriverWait(driver, 10).until(element_present) # Wait up to 3 seconds
        html = driver.page_source                        # Get the updated HTML
        soup = BeautifulSoup(html, "html.parser")

        data_elements1 = soup.find_all("div", class_="mb-1")
        data_elements2 = soup.find_all("div", class_= "text-neutral-200 text-base font-normal")
        
        for element1, element2 in zip(data_elements1, data_elements2):
            role = element1.text.strip()
            country = element2.text.strip()
            job_country_string += f"Job:{role}, Country:{country}"+"|" 
        
        return job_country_string
    
    except Exception as e:
        print(f"Error waiting for element: {e}")
    finally:
        driver.quit()



# --- 1. Your Input Data String ---
# Assuming your long string has each record on a new line,
# and fields are separated by a pipe '|'.
long_string_data = ted_source_data('https://clickhouse.com/company/careers')


# --- 2. Parse the Input String into a List of Dictionaries ---
# This step converts your raw string into a more manageable Python structure.
records = long_string_data.strip().split('|')
parsed_data = []
for record in records:
    parts = record.split(',')
    if len(parts) == 2: # Ensure we have all three expected parts
        parsed_data.append({
            "job_position": parts[0].strip(),
            "country": parts[1].strip()
        })
    else:
        print(f"Warning: Skipping malformed record: {parts}")


# --- 3. Create LlamaIndex Documents from Parsed Data ---
# Each entry becomes a Document. The 'text' field is what the LLM primarily reads,
# and 'metadata' can store structured information.
documents = []
for i, entry in enumerate(parsed_data):
    doc_text = (
        f"Job Position: {entry['job_position']}, "
        f"Country: {entry['country']}"
    )
    documents.append(
        Document(
            text=doc_text,
            metadata={
                "job_position": entry['job_position'],
                "country": entry['country']
            },
            id_=f"doc_{i}" # Assign a unique ID to each document
        )
    )

# --- 4. Define Pydantic Models for the Desired Output Structure ---
# This is crucial for guiding the LLM to produce structured, machine-readable output.
class SummarizedJobEntry(BaseModel):
    job_position: str = Field(..., description="The title of the job position.")
    country: str = Field(..., description="The country where this job position is found.")
    count: int = Field(..., description="The number of occurrences of this job position in the specified country.")

class JobSummary(BaseModel):
    jobs: List[SummarizedJobEntry] = Field(
        ...,
        description="A list of summarized job entries, grouped by job position and country, with their respective counts."
    )

# --- 5. Initialize the LLM ---
# Using gpt-3.5-turbo for this example. Set temperature to 0 for more deterministic output.
llm = OpenAI(model="gpt-4.1-nano", temperature=0)

# --- 6. Initialize the Pydantic Output Parser ---
# This parser will attempt to convert the LLM's raw text response into our Pydantic object.
parser = PydanticOutputParser(output_cls=JobSummary)
# parser = PydanticOutputParser(pydantic_schema=JobSummary)


format_instructions_str = ""
schema_dict = SummarizedJobEntry.model_json_schema()
format_instructions_str = json.dumps(schema_dict, indent=2)


# --- 7. Create a Custom Prompt Template for Aggregation ---
# The prompt is critical for instructing the LLM to perform the grouping and counting.
# `parser.format_instructions()` automatically injects the Pydantic schema into the prompt.
qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information, please summarize the job positions.
You need to perform an aggregation. Group the jobs by their 'job_position' and 'country'.
For each unique combination of 'job_position' and 'country', count how many times it appears.
Do not change the job position or country names, just use them as they are.
Your output MUST be a JSON object that strictly adheres to the following Pydantic schema:

{format_instructions}

Ensure the 'count' field is an integer.
"""
print(parser.get_format_string())

qa_prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str,
    template_var_mappings={"format_instructions": parser.get_format_string()},
)
# template_var_mappings={"format_instructions": format_instructions_str},


# --- 8. Build the LlamaIndex Index ---
# This creates a searchable index from your documents.
index = VectorStoreIndex.from_documents(documents, llm=llm)

# --- 9. Configure and Create the Query Engine ---
# For aggregation over all data, we need to ensure the retriever provides all documents
# to the LLM for processing. `similarity_top_k=len(documents)` ensures this.
retriever = index.as_retriever(similarity_top_k=len(documents))

# Configure the response synthesizer to use our custom prompt.
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    text_qa_template=qa_prompt_tmpl,
    llm=llm
)

# Create the query engine that combines retrieval and response synthesis.
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

# --- 10. Define and Execute the Query ---
# The query itself can be simple, as the aggregation logic is primarily in the prompt.
query = "Summarize the job data by job position and country, including counts in a table format."

# Query the index
response = query_engine.query(query)

# --- 11. Process and Display the Output ---
print("Raw LLM Response Object (contains parsed Pydantic object):")
# print(response)


try:
    # Parse the JSON string into a JobSummary object
    job_summary = JobSummary.model_validate_json(response.response)

    print("\n--- Parsed Pydantic Object (JobSummary) ---")
    # You can access the data directly from the Pydantic object
    # print(job_summary.model_dump_json(indent=2))  # Pretty print the JSON

    print("\n--- Summarized Table Output ---")
    print(f"{'Job Position':<65} | {'Count':>5} | {'Country':>30}")
    print("-" * 110)
    for job in job_summary.jobs:
        print(f"{job.job_position:<65} | {job.count:>5} | {job.country:>30}")


    print  ("\n--- Summary Statistics ---")
    total_jobs = len(job_summary.jobs)
    total_count = sum(job.count for job in job_summary.jobs)
    print(f"Total unique job positions: {total_jobs}")
    print(f"Total job count across all positions: {total_count} \n\n")
except Exception as e:
    print("\nError: Response was not parsed into the expected Pydantic object.")
    print(f"Error details: {e}")
    print("Raw response content:", job.job_position)



# # Check if the response was successfully parsed into our Pydantic object
# if isinstance(response.response, JobSummary):
#     print("\n--- Parsed Pydantic Object (JobSummary) ---")
#     # You can access the data directly from the Pydantic object
#     print(response.response.model_dump_json(indent=2)) # Pretty print the JSON

#     print("\n--- Summarized Table Output ---")
#     print(f"{'Job Position':<25} | {'Count':<5} | {'Country':<20}")
#     print("-" * 55)
#     for entry in response.response.summary:
#         print(f"{entry.job_position:<25} | {entry.count:<5} | {entry.country:<20}")
# else:
#     print("\nError: Response was not parsed into the expected Pydantic object.")
#     print("Raw response content:", response.response)