# =============================================================================
# AgentWorkflow with handler having context
# Created: 21, May 2025
# Updated: 21, May 2025
# Writer: Ted, Jung
# Description: How to handle the context with event in AgentWorkflow
#     Various event while running in a workflow.
# =============================================================================
import asyncio
import os
from xml.sax import handler


from git import WorkTreeRepositoryUnsupported
from tavily import AsyncTavilyClient
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import (
    Context, 
    JsonPickleSerializer, 
    JsonSerializer,
    InputRequiredEvent,
    HumanResponseEvent,
)

from llama_index.core.agent.workflow import (
    FunctionAgent,
    AgentWorkflow,
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCall,
    ToolCallResult,
)


async def set_name(ctx: Context, name: str) -> str:
    state = await ctx.set("state")
    state["name"] = name
    await ctx.set("state", state)
    return f"Name set to {name}"


llm = OpenAI(model="gpt-4.1-nano")
ta_api_key = os.environ["TAVILY_API_KEY"]



async def dangerous_task(ctx: Context) -> str:
    """A dangerous task that requires human confirmation."""

    # Human in the loop
    question = "Are you sure you want to proceed?"
    response = await ctx.wait_for_event(
        HumanResponseEvent,
        waiter_id=question,
        waiter_event=InputRequiredEvent(
            prefix=question,
            user_name="Ted",
        ),
        requirements={"user_name": "Ted"},
    )
    if response.response == "yes":
        return "Dangerous task completed successfully."
    else:
        return "Dangerous task aborted."



async def search_web(query: str) -> str:
    """Useful for using the web to answer questions"""
    client = AsyncTavilyClient(api_key=ta_api_key)
    return str(await client.search(query))


workflow = AgentWorkflow.from_tools_or_functions(
    [search_web],
    llm=llm,
    system_prompt="You are a helpful assistant that can search the web for information.",
)

workflow2 = AgentWorkflow.from_tools_or_functions(
    [set_name],
    llm=llm,
    system_prompt="You are a helpful assistant that can set a name.",
    initial_state={"name":"unset"},
)


workflow3 = AgentWorkflow.from_tools_or_functions(
    [dangerous_task],
    llm=llm,
    system_prompt="You are a helpful assistant that can perform dangerous task.",
)


workflow4 = AgentWorkflow.from_tools_or_functions(
    [dangerous_task],
    llm=llm,
    system_prompt="You are a helpful assistant that can perform dangerous task.",
)


# agent1 = FunctionAgent(name="agent1",tools=None)
# agent2 = FunctionAgent(name="agent2",tools=None)
# workflow4 = AgentWorkflow([agent1, agent2])

# The context is serializable, so it can be saved to a database, file, etc. and loaded back in later.
# Two serializer: JsonSerializer, JsonPickleSerializer (for object)
ctx = Context(workflow)
ctx_dict = ctx.to_dict(serializer=JsonSerializer())

restored_ctx = Context.from_dict(
    workflow, ctx_dict, serializer=JsonSerializer()
)



async def main():
    # response = await workflow.run(user_msg="What is the weather in Seoul?",)
    # print(str(response))

    # response = await workflow.run(user_msg="My name is Ted, nice to meet you", ctx=ctx)
    # print(str(response))

    # response = await workflow.run(user_msg="What is my name?", ctx=ctx)
    # print(str(response))

    # response = await workflow.run(user_msg="Do you still remember my name?", ctx=restored_ctx)

    # handler = workflow.run(user_msg="What is the weather in Tokyo?")

    # async for event in handler.stream_events():
    #     if isinstance(event, AgentStream):
    #         print(event.delta, end="", flush=True)# the current full response
    #         print(event.raw)
    #         print(event.current_agent_name)
    #     elif isinstance(event, AgentInput):
    #         print(event.input)                    # the current input messages
    #         print(event.current_agent_name)       # the current agent name
    #     elif isinstance(event, AgentOutput):
    #         print(event.response)                 # the current full response
    #         print(event.tool_calls)               # the selected tool calls, if any
    #         print(event.raw)                      # the raw llm api response
    #     elif isinstance(event, ToolCallResult):
    #         print(event.tool_name)
    #         print(event.tool_kwargs)
    #         print(event.tool_output)
    #     elif isinstance(event, ToolCall):
    #         print(event.tool_name)
    #         print(event.tool_kwargs)


    # ctx = Context(workflow)

    # response = await workflow2.run(user_msg="My name is Ted", ctx=ctx)
    # print(str(response))

    # ctx2 = Context(workflow2)

    # name = (await ctx2.get("state"))["name"]
    # print(name)

    ctx = Context(workflow3)

    # it returns coroutine object
    # it handle the execution of workflow
    handler = workflow3.run(
        user_msg="I want to proceed with the dangerous task.", ctx=ctx
    )


    # Human in the loop
    # handler: do two tasks
    # use context to send back response to tool
    # 1. event streaming (asynchronous streaming)
    # 2. gather the results
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            response = input(event.prefix).strip().lower()
            handler.ctx.send_event(
                HumanResponseEvent(
                    response=response,
                    user_name=event.user_name,
                )
            )
    # wait until to get the final output
    response = await handler
    print(str(response))



    # JsonSerializer (when to use?)
    handler = workflow4.run(user_msg="I want to proceed with the dangerous task.")
    
    input_ev = None
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            input_ev = event
            break

    
    # save the context somewhere for later
    ctx_dict = handler.ctx.to_dict(serializer=JsonSerializer())


    # get the response from the user after 1 hour
    response_str = input(input_ev.prefix).strip().lower()


    # restore the workflow
    restored_ctx = Context.from_dict(workflow4, ctx_dict, serializer=JsonSerializer())
    handler = workflow4.run(ctx=restored_ctx)
    handler.ctx.send_event(
        HumanResponseEvent(
            response=response_str,
            user_name=input_ev.user_name,
        )
    )

    response = await handler
    print(str(response))



asyncio.run(main())