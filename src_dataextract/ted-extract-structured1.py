# ===========================================================================
# Data Extraction: Structured Data
# Created: 14, Mar 2025
# Updated: 14, Mar 2025
# Writer: Ted, Jung
# Description: 
#   
# ===========================================================================

import json 

from pprint import pprint
from datetime import datetime
from pathlib import Path

from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
) 

from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader


class LineItem(BaseModel):
    """A line item in an invoice."""

    item_name: str = Field(description="The name of this item")
    price: float = Field(description="The price of this item")


class Invoice(BaseModel):
    """A representation of information from an invoice."""

    invoice_id: str = Field(
        description="A unique identifier for this invoice, often a number"
    )
    date: datetime = Field(description="The date this invoice was created")
    line_items: list[LineItem] = Field(
        description="A list of all the items in this invoice"
    )



pdf_reader = PDFReader()
documents = pdf_reader.load_data(file=Path("./uber_receipt.pdf"))
text = documents[0].text


# Important here
# llm itself be turned into structured llm having pydandic object(Invoice)
# Invoice has three fields(invoice_id, date, line_items)
# it works with given text and foaming output with structure(invoice)
llm = OpenAI(model="gpt-4o")
sllm = llm.as_structured_llm(Invoice)
response = sllm.complete(text)

json_response = json.loads(response.text)
print(json.dumps(json_response, indent=2))


pprint(response.raw)