# ===========================================================================
# Working example of a workflow with parallel events
# Created: 13, Mar 2025
# Updated: 13, Mar 2025
# Writer: Ted, Jung
# Description: 
#   handle the concurrent and parallel events
# ===========================================================================


import asyncio
import random 

from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.workflow import (
    Workflow, 
    Context, 
    StartEvent, 
    StopEvent, 
    step,
    Event
)
from llama_index.core.response_synthesizers import get_response_synthesizer


class FirstEvent(Event):
    query: str

class SecondEvent(Event):
    query: str

class ThirdEvent(Event):
    query: str

class FinalEvent(Event):
    query: str


class ParallelFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> FirstEvent | SecondEvent | ThirdEvent:
        ctx.send_event(FirstEvent(query="Query 1"))
        ctx.send_event(SecondEvent(query="Query 2"))
        ctx.send_event(ThirdEvent(query="Query 3-1"))
        ctx.send_event(ThirdEvent(query="Query 3-2"))


    @step
    async def evt_first(self, ctx: Context, ev: FirstEvent) -> FinalEvent:
        print("Running slow query ", ev.query)
        await asyncio.sleep(random.randint(1, 2))

        return FinalEvent(query=ev.query)


    @step
    async def evt_second(self, ctx: Context, ev: SecondEvent) -> FinalEvent:
        print("Running slow query ", ev.query)
        await asyncio.sleep(random.randint(6, 7))

        return FinalEvent(query=ev.query)


    @step
    async def evt_third(self, ctx: Context, ev: ThirdEvent) -> FinalEvent:
        print("Running slow query ", ev.query)
        await asyncio.sleep(random.randint(2, 3))

        return FinalEvent(query=ev.query)
    

    @step
    async def synthesize(self, ctx: Context, ev: FinalEvent) -> StopEvent | None:
        data = ctx.collect_events(ev, [FinalEvent]*4)
        if data is None:
            return None

        first_evt, second_evt, third_evt, third_evt = data

        # Convert strings to TextNode objects
        nodes_with_scores = [
            NodeWithScore(node=TextNode(text=first_evt.query)),
            NodeWithScore(node=TextNode(text=second_evt.query)),
            NodeWithScore(node=TextNode(text=third_evt.query)),
            NodeWithScore(node=TextNode(text=third_evt.query)),
        ]


        synthesizer = get_response_synthesizer(response_mode="compact")
        response = synthesizer.synthesize("what is the keyword?", nodes=nodes_with_scores)
        ev.query = response

        print("Synthesizing results ", ev.query)
        return StopEvent(result=ev.query)



async def main():
    p = ParallelFlow(timeout=20, verbose=True)
    await p.run()


if __name__ == "__main__":
    asyncio.run(main())

