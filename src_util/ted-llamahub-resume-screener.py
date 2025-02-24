# ===========================================================================
# Resume screening
# Created: 23, Feb 2025
# Updated: 23, Feb 2025
# Writer: Ted, Jung
# Description:
#   from LlamaHub (WikipediaReader, ResumeScreenPack)
#   -> need to set False of refresh cache (gpt-4 is default , cost highly)
# ===========================================================================



# Do sentence splitting on the first piece of text
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.llama_pack import download_llama_pack

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5") 
Settings.llm = OpenAI(model="gpt-4o-mini")
llm = OpenAI(model="gpt-4o-mini")


# wikipediaReader, it returns List
loader = WikipediaReader()
documents = loader.load_data(
    pages=["OpenAI", "Sam Altman", "Mira Murati", "Emmett Shear"],
    auto_suggest=False,
)


sentence_splitter = SentenceSplitter(chunk_size=1024)

# Get the first 1024 tokens for each entity
openai_node = sentence_splitter.get_nodes_from_documents([documents[0]])[0]
sama_node = sentence_splitter.get_nodes_from_documents([documents[1]])[0]
mira_node = sentence_splitter.get_nodes_from_documents([documents[2]])[0]
emmett_node = sentence_splitter.get_nodes_from_documents([documents[3]])[0]


# Download a pack from LlamaHub
ResumeScreenerPack = download_llama_pack(
    "ResumeScreenerPack", 
    "./resume_screener_pack",
    refresh_cache=False
)


# Job description
meta_jd = """\
Meta is embarking on the most transformative change to its business and technology in company history, and our Machine Learning Engineers are at the forefront of this evolution. By leading crucial projects and initiatives that have never been done before, you have an opportunity to help us advance the way people connect around the world.
 
The ideal candidate will have industry experience working on a range of recommendation, classification, and optimization problems. You will bring the ability to own the whole ML life cycle, define projects and drive excellence across teams. You will work alongside the world’s leading engineers and researchers to solve some of the most exciting and massive social data and prediction problems that exist on the web.\
"""


# JD + Conditions
# Read PDF
resume_screener = ResumeScreenerPack(
    job_description=meta_jd,
    criteria=[
        "2+ years of experience in one or more of the following areas: machine learning, recommendation systems, pattern recognition, data mining, artificial intelligence, or related technical field",
        "Experience demonstrating technical leadership working with teams, owning projects, defining and setting technical direction for projects",
        "Bachelor's degree in Computer Science, Computer Engineering, relevant technical field, or equivalent practical experience.",
    ],
)
response = resume_screener.run(resume_path="Ted-Resume.pdf")

for cd in response.criteria_decisions:
    print("### CRITERIA DECISION")
    print(cd.reasoning)
    print(cd.decision)
print("#### OVERALL REASONING ##### ")
print(str(response.overall_reasoning))
print(str(response.overall_decision))


# Comparision
resume_screener = ResumeScreenerPack(
    job_description="We're looking to hire a front-end engineer",
    criteria=[
        "The individual needs to be experienced in front-end / React / Typescript"
    ],
)
response = resume_screener.run(resume_path="Ted-Resume.pdf")
print(str(response.overall_reasoning))
print(str(response.overall_decision))


# Job Description
job_description = f"""\
We're looking to hire a CEO for OpenAI.

Instead of listing a set of specific criteria, each "criteria" is instead a short biography of a previous CEO.\

For each criteria/bio, outline if the candidate's experience matches or surpasses that of the candidate.

Also, here's a description of OpenAI from Wikipedia: 
{openai_node.get_content()}
"""

profile_strs = [
    f"Profile: {n.get_content()}" for n in [sama_node, mira_node, emmett_node]
]


resume_screener = ResumeScreenerPack(
    job_description=job_description, criteria=profile_strs
)

response = resume_screener.run(resume_path="Ted-Resume.pdf")

for cd in response.criteria_decisions:
    print("### CRITERIA DECISION")
    print(cd.reasoning)
    print(cd.decision)
print("#### OVERALL REASONING ##### ")
print(str(response.overall_reasoning))
print(str(response.overall_decision))