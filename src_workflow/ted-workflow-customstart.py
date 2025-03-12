# ===========================================================================
# Workflow with custom start event
# Created: 12, Mar 2025
# Updated: 12, Mar 2025
# Writer: Ted, Jung
# Description: 
#   Be able to use custome StartEvent with some trick like handle_start
# ===========================================================================


import asyncio
import os
import logging

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from llama_index.llms.openai import OpenAI
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Inherit start event and stop event
class MyCustomStartEvent(StartEvent):
    a_string_field: str
    a_path_to_somewhere: Path
    an_index: VectorStoreIndex
    an_llm: OpenAI


class MyStopEvent(StopEvent):
    critique: str


class JokeEvent(Event):
    joke: str


class JokeFlow(Workflow):
    llm = OpenAI(model="gpt-4o-mini", timeout=10.0)


    @step
    async def handle_start(self, ev: StartEvent) -> MyCustomStartEvent:
        return ev


    @step
    async def generate_joke_from_index(self, ev: MyCustomStartEvent) -> JokeEvent:
        query_engine = ev.an_index.as_query_engine(llm=ev.an_llm)
        try:
            topic = query_engine.query(
                f"What is the closest topic to {ev.a_string_field}"
            )
            prompt = f"Write your best joke about {topic}."
            response = await ev.an_llm.acomplete(prompt)
            ev.a_path_to_somewhere.write_text(str(response))
            return JokeEvent(joke=str(response))
        except Exception as e:
            logger.error(f"Error generating joke: {e}")
            return JokeEvent(joke=f"Error: {e}")
        

    @step
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke
        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        try:
            response = await self.llm.acomplete(prompt)
            return StopEvent(result={"critique":str(response)})
        except Exception as e:
            logger.error(f"Error critiquing joke: {e}")
            return StopEvent(result={"critique":f"Error: {e}"})




async def main():
    curr_dir = os.getcwd()
    reader = SimpleDirectoryReader(input_files = [f"{curr_dir}/data/paul_graham/paul_graham_essay_short.txt"])
    documents = reader.load_data()
    ted_index = VectorStoreIndex.from_documents(documents)
    ted_llm = OpenAI(model="gpt-4o-mini", timeout=10.0)
    ted_str = "Paul Graham essay on What I Worked On"
    ted_path = Path(f"{curr_dir}/data/paul_graham/paul_graham_essay_short.txt")

    custom_start_event = MyCustomStartEvent(
        a_string_field=ted_str,
        a_path_to_somewhere=ted_path,
        an_index=ted_index,
        an_llm=ted_llm
    )

    w = JokeFlow(timeout=60, verbose=False)
    result = await w.run(start_event=custom_start_event)
    print(str(result))

asyncio.run(main())