# ===========================================================================
# Resume screening with AI
# Created: 23, Feb 2025
# Updated: 23, Feb 2025
# Writer: Ted, Jung
# Description:
#   from LlamaHub (WikipediaReader, ResumeScreenPack)
#   -> need to set False of refresh cache (gpt-4 is default , cost highly)
# ===========================================================================


import streamlit as st

# Do sentence splitting on the first piece of text
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.llama_pack import download_llama_pack

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5") 
# Settings.llm = OpenAI(model="gpt-4o-mini")
llm = OpenAI(model="gpt-4o-mini")


# wikipediaReader, it returns List with data
wiki_loader = WikipediaReader()
documents = wiki_loader.load_data(
    pages=["OpenAI", "Sam Altman", "Mira Murati", "Emmett Shear"],
    auto_suggest=False,
)


sentence_splitter = SentenceSplitter(chunk_size=1024)

# Get the first 1024 tokens for each entity
# Turn document into a list of nodes.
# reason why we use sentence splitter is that the document is too long
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


# Job Description
meta_jd = """\
We are looking for a technically savvy and business-minded solutions architect to deeply partner with our most strategic and high-impact platform customers, guiding them through application ideation, development, delivery, and scale to accelerate and maximize the value of what they build with our platform. You will have the opportunity to work on the most novel and creative use cases being built on our API, serving as a critical partner for collecting and delivering high fidelity feedback to Product and Research teams.\
"""


# Criterias for screening for the job
# Read PDF
# Position for CA, SA
st.write("# Position for CA, SA! 👋")
resume_screener = ResumeScreenerPack(
    job_description=meta_jd,
    criteria=[
        "Deeply embed with our most strategic platform customers, serving as their technical thought partner in ideating and building novel applications on our API.",
        "Proactively provide guidance to our customers on how to maximize business impact from their applications, accelerating their time to value.",
        "Experiment and prototype solutions with and for your customers.",
        "Forge and manage relationships with our customers’ leadership and stakeholders to ensure their application’s successful deployment and scale.",
        "Contribute to our open-source developer and enterprise resources.",
        "Scale the Solutions Architect function through sharing knowledge, codifying best practices, and publishing notebooks to our internal and external repositories.",
        "Validate, synthesize, and deliver high-signal feedback to the Product and Research teams.",
        "Use your expertise in programming with Python and Javascript.",
        "Have 5+ years of technical consulting (or equivalent) experience.",
        "Are proficient in Python and Javascript.",
        "Built and/or delivered prototypes on top of our API platform.",
        "Led complex technical projects and programs with many stakeholders.",
        "Can proactively identify opportunities for maximizing our customers’ business value through leveraging the OpenAI API.",
        "Own problems end-to-end, and are willing to pick up whatever knowledge you're missing to get the job done to ensure both your team and our customers succeed.",
        "Have a humble attitude and an eagerness to help others with empathy.",
        "Operate with high horsepower, are adept at frequent context switching and working on multiple projects at once with expansive ownership, and ruthlessly prioritize.",
        "Thrive in dynamic environments and can navigate ambiguity with ease.",
    ],
    llm=llm
)
response = resume_screener.run(resume_path="/Users/tedj/Ted-person/Ted Personal/resume/OpenAI/Ted-Resume.pdf")

for cd in response.criteria_decisions:
    st.write("##### CRITERIA DECISION")
    st.write(cd.reasoning)
    st.write(cd.decision)

st.write("#### OVERALL REASONING")
st.write(str(response.overall_reasoning))
st.write(str(response.overall_decision))


# Comparision of Specific profiles
# st.write("# Position for Front-end Engineer! 👋")
# resume_screener = ResumeScreenerPack(
#     job_description="We're looking to hire a front-end engineer",
#     criteria=[
#         "The individual needs to be experienced in front-end / React / Typescript"
#     ],
#     llm=llm
# )
# response = resume_screener.run(resume_path="/Users/tedj/Ted-person/Ted Personal/resume/OpenAI/Ted-Resume.pdf")
# for cd in response.criteria_decisions:
#     st.write("##### CRITERIA DECISION")
#     st.write(cd.reasoning)
#     st.write(cd.decision) 

# st.write("#### OVERALL REASONING")
# print(str(response.overall_reasoning))
# print(str(response.overall_decision))



# Job Description
# st.write("# Position for CEO! 👋")
# job_description = f"""\
# We're looking to hire a CEO for OpenAI.

# Instead of listing a set of specific criteria, each "criteria" is instead a short biography of a previous CEO.\

# For each criteria/bio, outline if the candidate's experience matches or surpasses that of the candidate.

# Also, here's a description of OpenAI from Wikipedia: 
# {openai_node.get_content()}
# """

# profile_strs = [
#     f"Profile: {n.get_content()}" for n in [sama_node, mira_node, emmett_node]
# ]


# resume_screener = ResumeScreenerPack(
#     job_description=job_description, 
#     criteria=profile_strs,
#     llm=llm
# )

# response = resume_screener.run(resume_path="/Users/tedj/Ted-person/Ted Personal/resume/OpenAI/Ted-Resume.pdf")

# for cd in response.criteria_decisions:
#     st.write("### CRITERIA DECISION")
#     st.write(cd.reasoning)
#     st.write(cd.decision)


# st.write("#### OVERALL REASONING")
# st.write(str(response.overall_reasoning))
# st.write(str(response.overall_decision))

