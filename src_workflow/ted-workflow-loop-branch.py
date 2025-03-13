# ===========================================================================
# Workflow having loops and braches
# Created: 13, Mar 2025
# Updated: 13, Mar 2025
# Writer: Ted, Jung
# Description: 
#   handle the concurrent and parallel events
#   events to events in steps
#   maintain the state of the workflow using events
# ===========================================================================


import asyncio
import random


from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
from llama_index.llms.openai import OpenAI



class FailedEvent(Event):
    error: str

class QueryEvent(Event):
    query: str

class LoopExampleFlow(Workflow):
    @step
    async def answer_query(self, ev: StartEvent | QueryEvent) -> FailedEvent | StopEvent:

        if hasattr(ev, "query"):
            query = ev.query

            llm = OpenAI(model="gpt-4o-mini", timeout=300.0)
            response = await llm.acomplete(query)
            print(response)

            return StopEvent(result="The answer to your query")
        
        elif hasattr(ev, "data"):
            return FailedEvent(error="No query provided.")


    @step
    async def improve_query(self, ev: FailedEvent) -> QueryEvent | StopEvent:
        # improve the query or decide it can't be fixed
        random_number = random.randint(0, 1)
        if random_number == 0:
            return QueryEvent(query="Here's a better query.")
        else:
            return StopEvent(result="Your query can't be fixed.")




async def ted_main():
    w = LoopExampleFlow(timeout=10, verbose=False)
    result = await w.run(query="What's LlamaIndex?")
    print(result)

    result = await w.run(data="data")
    print(result)


if __name__ == "__main__":
    asyncio.run(ted_main())