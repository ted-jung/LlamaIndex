# ===========================================================================
# Data Extraction2: Structured Data
# Created: 14, Mar 2025
# Updated: 14, Mar 2025
# Writer: Ted, Jung
# Description: 
#   structured prediction uses function calling to extract structured outputs
# ===========================================================================

import os
import json

from datetime import datetime
from pathlib import Path

from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
) 

from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.prompts import PromptTemplate



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
    payment: str = Field(
        description="A card name and with number and date"
    )
    line_items: list[LineItem] = Field(
        description="A list of all the items in this invoice"
    )


curr_dir = os.getcwd()

pdf_reader = PDFReader()
documents = pdf_reader.load_data(file=Path(f"{curr_dir}/data/pdf/uber_receipt.pdf"))

text = documents[0].text


prompt = PromptTemplate(
    """Extract an invoice from the following text. If you cannot find an invoice ID, 
       use the company name '{company_name}' and the date as the invoice ID: {text}
    """
)
llm = OpenAI(model="gpt-4o-mini")
response = llm.structured_predict(
    Invoice, prompt, text=text, company_name="Uber"
)

json_output = response.model_dump_json()
print(json.dumps(json.loads(json_output), indent=2))