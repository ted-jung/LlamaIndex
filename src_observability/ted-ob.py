# =============================================================================
# Obserability and Monitoring using instrumentation
# Created: 18, Jun 2025
# Updated: 18, Jun 2025
# Writer: Ted, Jung
# Description: Instrumentation for event/eventhandler and span/spanhandler
#              with dispatcher
# =============================================================================

import llama_index.core.instrumentation as instrument
import inspect

from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from typing import Any, Dict, Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation.span.base import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.instrumentation.span import BaseSpan
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler



Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = OpenAI(model="gpt-4.1-nano", request_timeout=360.0)


class MyEventHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "MyEventHandler"

    def handle(self, event) -> None:
        """Logic for handling event."""
        # THIS IS WHERE YOU ADD YOUR LOGIC TO HANDLE EVENTS
        print(event.dict())
        print("")
        with open("log.txt", "a") as f:
            f.write(str(event))
            f.write("\n")



class MyCustomSpan(BaseSpan):
    custom_field_1: Any = Field(...)
    custom_field_2: Any = Field(...)


class MyCustomSpanHandler(BaseSpanHandler[MyCustomSpan]):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "MyCustomSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[MyCustomSpan]:
        """Create a span."""
        # logic for creating a new MyCustomSpan
        pass

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to exit a span."""
        pass

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to drop a span."""
        pass

dispatcher = instrument.get_dispatcher()

span_handler = SimpleSpanHandler()

dispatcher.add_event_handler(MyEventHandler())
dispatcher.add_span_handler(span_handler)

qe_dispatcher = instrument.get_dispatcher(
    "llama_index.core.base.base_query_engine"
)



documents = SimpleDirectoryReader(input_dir="./data/paul_graham").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

query_result = query_engine.query("Who is Paul?")

print(query_result)