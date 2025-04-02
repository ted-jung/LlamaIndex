# =============================================================================
# Basic prompt
# Date: 20, Jan 2025
# Updated: 2, Apr 2025
# Writer: Ted, Jung
# Description: It is important to clarify on user's question and make exact answer
#              There are a few examples regarding handling of prompting
# =============================================================================
 
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=720.0)
Settings.llm = llm



# 1. It is a partial prompt formatting
qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{{context_str}}
---------------------
Given the context information and not prior knowledge, answer the query.
Please write the answer in the style of {{tone_name}}
Query: {{query_str}}
Answer: \
"""

prompt_tmpl = RichPromptTemplate(qa_prompt_tmpl_str)
partial_prompt_tmpl = prompt_tmpl.partial_format(tone_name="Shakespeare")

fmt_prompt = partial_prompt_tmpl.format(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
print(fmt_prompt)


fmt_prompt = partial_prompt_tmpl.format_messages(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
print(fmt_prompt)



# 2. Prompt template and variable mapping
# NOTE: here notice we use `my_context` and `my_query` as template variables

qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{{my_context}}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {{my_query}}
Answer: \
"""

template_var_mappings = {"context_str": "my_context", "query_str": "my_query"}

prompt_tmpl = RichPromptTemplate(
    qa_prompt_tmpl_str, 
    template_var_mappings=template_var_mappings
)

fmt_prompt = prompt_tmpl.format(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
print(fmt_prompt)



# 3. Prompt Function Mapping
qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \
"""


def format_context_fn(**kwargs):
    # format context with bullet points
    context_list = kwargs["context_str"].split("\n\n")
    fmtted_context = "\n\n".join([f"- {c}" for c in context_list])
    return fmtted_context


prompt_tmpl = RichPromptTemplate(
    qa_prompt_tmpl_str, function_mappings={"context_str": format_context_fn}
)

context_str = """\
In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters.

Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases.

Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models.
"""

fmt_prompt = prompt_tmpl.format(
    context_str=context_str, query_str="How many params does llama 2 have"
)
print(fmt_prompt)




# 4. Dynamic few-shot examples
# index -> retrieve -> few shot example
text_to_sql_prompt_tmpl_str = """\
You are a SQL expert. You are given a natural language query, and your job is to convert it into a SQL query.

Here are some examples of how you should convert natural language to SQL:

{{ examples }}


Now it's your turn.

Query: {{ query_str }}
SQL: 
"""


# Setup few-shot examples
example_nodes = [
    TextNode(
        text="Query: How many params does llama 2 have?\nSQL: SELECT COUNT(*) FROM llama_2_params;"
    ),
    TextNode(
        text="Query: How many layers does llama 2 have?\nSQL: SELECT COUNT(*) FROM llama_2_layers;"
    ),
]

# Create index & retriever
index = VectorStoreIndex(nodes=example_nodes)
retriever = index.as_retriever(similarity_top_k=1)


def get_examples_fn(**kwargs):
    query = kwargs["query_str"]
    examples = retriever.retrieve(query)
    return "\n\n".join(node.text for node in examples)


prompt_tmpl = RichPromptTemplate(
    text_to_sql_prompt_tmpl_str,
    function_mappings={"examples": get_examples_fn},
)

prompt = prompt_tmpl.format(
    query_str="What are the number of parameters in the llama 2 model?"
)
print(prompt)


