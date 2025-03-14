# ===========================================================================
# Workflow - Human in the loop
# Created: 13, Mar 2025
# Updated: 14, Mar 2025
# Writer: Ted, Jung
# Description: 
#   Sometimes, Need to go through a process that human intervention in the loop
#   where making a decision based on his experiences.
# ===========================================================================


import uuid
import asyncio

from typing import Optional, List

from llama_index.llms.openai import OpenAI
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.prompts import PromptTemplate

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

class Segment(BaseModel):
    """Data model for generating segments of a story."""

    plot: str = Field(
        description="""
            The plot of the adventure for the current segment. 
            The plot should be no longer than 3 sentences.
        """
    )
    actions: List[str] = Field(
        default=[],
        description="""
            The list of actions the protaganist can take 
            that will shape the plot and actions of the next segment.
        """,
    )


SEGMENT_GENERATION_TEMPLATE = """
You are working with a human to create a story in the style of choose your own adventure.

The human is playing the role of the protaganist in the story which you are tasked to
help write. To create the story, we do it in steps, where each step produces a BLOCK.
Each BLOCK consists of a PLOT, a set of ACTIONS that the protaganist can take, and the
chosen ACTION. 

Below we attach the history of the adventure so far.

PREVIOUS BLOCKS:
---
{running_story}

Continue the story by generating the next block's PLOT and set of ACTIONs. If there are
no previous BLOCKs, start an interesting brand new story. Give the protaganist a name(Big Ted) and an
interesting challenge to solve.


Use the provided data model to structure your output.
"""


FINAL_SEGMENT_GENERATION_TEMPLATE = """
You are working with a human to create a story in the style of choose your own adventure.

The human is playing the role of the protaganist in the story which you are tasked to
help write. To create the story, we do it in steps, where each step produces a BLOCK.
Each BLOCK consists of a PLOT, a set of ACTIONS that the protaganist can take, and the
chosen ACTION. Below we attach the history of the adventure so far.

PREVIOUS BLOCKS:
---
{running_story}

The story is now coming to an end. With the previous blocks, wrap up the story with a
closing PLOT. Since it is a closing plot, DO NOT GENERATE a new set of actions.

Use the provided data model to structure your output.
"""


# Let's see an example segment
llm = OpenAI("gpt-4o-mini")
segment = llm.structured_predict(
    Segment,
    PromptTemplate(SEGMENT_GENERATION_TEMPLATE),
    running_story="",
)

print(segment)




BLOCK_TEMPLATE = """
BLOCK
===
PLOT: {plot}
ACTIONS: {actions}
CHOICE: {choice}
"""


# Block is a pydantic data model having a segment and a choice
# Segment is also a pydantic data model having a plot and actions
class Block(BaseModel):
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    segment: Segment
    choice: Optional[str] = None
    block_template: str = BLOCK_TEMPLATE

    def __str__(self):
        return self.block_template.format(
            plot=self.segment.plot,
            actions=", ".join(self.segment.actions),
            choice=self.choice or "",
        )


block = Block(segment=segment)
print(block)



class NewBlockEvent(Event):
    block: Block


class HumanChoiceEvent(Event):
    block_id: str


# Now create a next block to build a story based on the previous block using workflow
# The workflow will generate a segment and prompt the human to make a choice
# The human will then make a choice and the workflow will continue to the next step
# The workflow will continue until the story is complete
class ChooseYourOwnAdventureWorkflow(Workflow):
    def __init__(self, max_steps: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.llm = OpenAI("gpt-4o-mini")
        self.max_steps = max_steps

    @step
    async def create_segment(self, ctx: Context, ev: StartEvent | HumanChoiceEvent) -> NewBlockEvent | StopEvent:
        blocks = await ctx.get("blocks", [])
        print(blocks)
        running_story = "\n".join(str(b) for b in blocks)

        if len(blocks) < self.max_steps:
            new_segment = self.llm.structured_predict(
                Segment,
                PromptTemplate(SEGMENT_GENERATION_TEMPLATE),
                running_story=running_story,
            )
            new_block = Block(segment=new_segment)
            blocks.append(new_block)
            await ctx.set("blocks", blocks)
            return NewBlockEvent(block=new_block)
        else:
            final_segment = self.llm.structured_predict(
                Segment,
                PromptTemplate(FINAL_SEGMENT_GENERATION_TEMPLATE),
                running_story=running_story,
            )
            final_block = Block(segment=final_segment)
            blocks.append(final_block)
            return StopEvent(result=blocks)

    @step
    async def prompt_human(self, ctx: Context, ev: NewBlockEvent) -> HumanChoiceEvent:
        block = ev.block

        # get human input
        human_prompt = f"\n===\n{ev.block.segment.plot}\n\n"
        human_prompt += "Choose your adventure:\n\n"
        human_prompt += "\n".join(ev.block.segment.actions)
        human_prompt += "\n\n"
        human_input = input(human_prompt)

        blocks = await ctx.get("blocks")
        block.choice = human_input
        blocks[-1] = block
        await ctx.set("block", blocks)

        return HumanChoiceEvent(block_id=ev.block.id_)



async def main():
    w = ChooseYourOwnAdventureWorkflow(timeout=10, verbose=False)
    result = await w.run()

    final_story = "\n\n".join(b.segment.plot for b in result)
    print(final_story)


if __name__ == "__main__":
    asyncio.run(main()) 