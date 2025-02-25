# ===========================================================================
# Fine tuning 
# Updated: 25, Feb 2025
# Writer: Ted, Jung
# Description:  Generate Questions&Answers unsing LLM1
#               Evaluate the answer by another LLM2
# ===========================================================================


import os
import tqdm
import nest_asyncio
import asyncio


# wikipedia pages
from llama_index.readers.wikipedia import WikipediaReader


# generate questions against chunks
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama



from llama_index.llms.openai import OpenAI
from llama_index.finetuning.callbacks import OpenAIFineTuningHandler
from llama_index.core.callbacks import CallbackManager
from llama_index.core.evaluation import CorrectnessEvaluator



nest_asyncio.apply()


# Sourcing the cities documents from Wikipedia 
cities = [
    "San Francisco",
    # "Toronto",
    # "New York",
    # "Vancouver",
    # "Montreal",
    # "Tokyo",
    # "Singapore",
    # "Paris",
    # "Seoul"
]

documents = WikipediaReader().load_data(
    pages=[f"History of {x}" for x in cities]
)

QUESTION_GEN_PROMPT = (
    "You are a Teacher/ Professor. Your task is to setup "
    "a quiz/examination. Using the provided context, formulate "
    "a single question that captures an important fact from the "
    "context. Restrict the question to the context information provided."
)

gen_llm = OpenAI(model="gpt-4o-mini", temperature=0.3)
# gen_llm = Ollama(model="llama3.2", temperature=0.3, request_timeout=720.0)

# Instantiate a DatasetGenerator to generate Questions and Save it to be used later
# QueryResponseDataset
dataset_generator = DatasetGenerator.from_documents(
    documents,
    question_gen_query=QUESTION_GEN_PROMPT,
    llm=gen_llm,
    num_questions_per_chunk=10,
)
qrd = dataset_generator.generate_dataset_from_nodes(num=350)
curr_dir = os.getcwd()
qrd.save_json(f"{curr_dir}/src_finetuning/data/qrd.json")



# Generate answer to the Questions


# Create vector index using Wikipedia documents
# documents -> vector index -> retriver -> query engine (other llm) <- 65% questions for training
the_index = VectorStoreIndex.from_documents(documents=documents)

# Create the retriver on this index
the_retriever = VectorIndexRetriever(
    index=the_index,
    similarity_top_k=2,
)

# Use other LLM for querying
# If you do not permission for inferencing? then pay first to get accessible token
# You can use another LLM
# other_llm = HuggingFaceInferenceAPI(model_name="meta-llama/Llama-2-7b-chat-hf",context_window=2048)

other_llm = OpenAI(model="gpt-4o-mini", temperature=0.3)
query_engine = RetrieverQueryEngine.from_args(retriever=the_retriever, llm=other_llm)




# Will use 65% of the generated questions for training
# qrd.qr_pairs: 0-Question, 1-Answer
train_dataset = []
num_train_questions = int(0.65 * len(qrd.qr_pairs))

for q, a in tqdm.tqdm(qrd.qr_pairs[:num_train_questions]):
    # data for this q
    data_entry = {"question": q, "reference": a}
    response = query_engine.query(q)
    response_struct = {}
    response_struct["model"] = "gpt-4o-mini"
    response_struct["text"] = str(response)
    response_struct["context"] = (
        response.source_nodes[0].node.text[:1000] + "..."
    )

    data_entry["response_data"] = response_struct
    train_dataset.append(data_entry)



# Get GPT-4 evaluation on other llm(Llama3.2) Answer
# Instantiate the gpt-4 judge


finetuning_handler = OpenAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])
gpt_4o_mini = OpenAI(
    temperature=0, model="gpt-4o-mini", callback_manager=callback_manager
)

gpt4_judge = CorrectnessEvaluator(llm=gpt_4o_mini)

# for `training`
async def ted_evaluate():
    for data_entry in tqdm.tqdm(train_dataset):
        eval_result = await gpt4_judge.aevaluate(
            query=data_entry["question"],
            response=data_entry["response_data"]["text"],
            context=data_entry["response_data"]["context"],
            reference=data_entry["reference"],
        )

        # save final result
        judgement = {}
        judgement["llm"] = "gpt-4"
        judgement["score"] = eval_result.score
        judgement["text"] = eval_result.response
        data_entry["evaluations"] = [judgement]

    finetuning_handler.save_finetuning_events(f"{curr_dir}/src_finetuning/data/correction_finetuning_events2.jsonl")

if __name__ =="__main__":
    asyncio.run(ted_evaluate())