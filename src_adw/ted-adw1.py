# =============================================================================
# ADW(Agent Document Workflows)
# Created: 28, Apr 2025
# Updated: 28, Apr 2025
# Writer: Ted, Jung
# Description: Agentic workflow for compliance checking
#     1. define tools
#     2. define agents (tools + system prompt + llm + handoff)
#     3. define agent workflow (agents + root agent + initial state)
#     4. run the workflow
# =============================================================================


import asyncio
import nest_asyncio
import os
import json


from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
# from IPython.display import clear_output

from llama_parse import LlamaParse
# from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.openai import OpenAI

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
    draw_all_possible_flows
)

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole, LLM
from llama_index.core.retrievers import BaseRetriever

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse


curr_dir = os.getcwd()
nest_asyncio.apply()


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = OpenAI(model="gpt-4o-mini")


# Create an index and turn it into a retriever
# index = LlamaCloudIndex(
#     name="ted-adw1-idx",
#     project_name="ted-llama-project",
#     project_id=os.getenv("LLAMA_PROJECT_ID"),
# )
# parser = LlamaParse(result_type="markdown")
# file_extractor = {".pdf": parser}
# reader = SimpleDirectoryReader(input_files=[f"{curr_dir}/src_adw/data/gdpr.pdf"], file_extractor=file_extractor)
# documents = reader.load_data()



# Document read using a reader with a parser to build the index & retriever
# turn the pdf file into retriever to be used in the workflow for compliance checking
parser = LlamaParse(result_type="markdown")
reader = SimpleDirectoryReader(input_dir=f"{curr_dir}/src_adw/data", required_exts=[".pdf"])
documents = reader.load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=2)




# Define Output Schema(Data Structure) for ContractClause and ContractExtraction
# Convert raw data into python objects
class ContractClause(BaseModel):
    clause_text: str = Field(..., description="The exact text of the clause.")
    mentions_data_processing: bool = Field(False, description="True if the clause involves personal data collection or usage.")
    mentions_data_transfer: bool = Field(False, description="True if the clause involves transferring personal data, especially to third parties or across borders.")
    requires_consent: bool = Field(False, description="True if the clause explicitly states that user consent is needed for data activities.")
    specifies_purpose: bool = Field(False, description="True if the clause specifies a clear purpose for data handling or transfer.")
    mentions_safeguards: bool = Field(False, description="True if the clause mentions security measures or other safeguards for data.")

class ContractExtraction(BaseModel):
    vendor_name: Optional[str] = Field(None, description="The vendor's name if identifiable.")
    effective_date: Optional[str] = Field(None, description="The effective date of the agreement, if available.")
    governing_law: Optional[str] = Field(None, description="The governing law of the contract, if stated.")
    clauses: List[ContractClause] = Field(..., description="List of extracted clauses and their relevant indicators.")



# Define Check Schema for Compliance Checking
class GuidelineMatch(BaseModel):
    guideline_text: str = Field(..., description="The single most relevant guideline excerpt related to this clause.")
    similarity_score: float = Field(..., description="Similarity score indicating how closely the guideline matches the clause, e.g., between 0 and 1.")
    relevance_explanation: Optional[str] = Field(None, description="Brief explanation of why this guideline is relevant.")

class ClauseComplianceCheck(BaseModel):
    clause_text: str = Field(..., description="The exact text of the clause from the contract.")
    matched_guideline: Optional[GuidelineMatch] = Field(None, description="The most relevant guideline extracted via vector retrieval.")
    compliant: bool = Field(..., description="Indicates whether the clause is considered compliant with the referenced guideline.")
    notes: Optional[str] = Field(None, description="Additional commentary or recommendations.")



# Define Output Schema for Final
class ComplianceReport(BaseModel):
    vendor_name: Optional[str] = Field(None, description="The vendor's name if identified from the contract.")
    overall_compliant: bool = Field(..., description="Indicates if the contract is considered overall compliant.")
    summary_notes: Optional[str] = Field(None, description="General summary or recommendations for achieving full compliance.")




# Setup Contract Review Workflow

CONTRACT_EXTRACT_PROMPT = """\
You are given contract data below. \
Please extract out relevant information from the contract into the defined schema - the schema is defined as a function call.\

{contract_data}
"""

CONTRACT_MATCH_PROMPT = """\
Given the following contract clause and the corresponding relevant guideline text, evaluate the compliance \
and provide a JSON object that matches the ClauseComplianceCheck schema.

**Contract Clause:**
{clause_text}

**Matched Guideline Text(s):**
{guideline_text}
"""


COMPLIANCE_REPORT_SYSTEM_PROMPT = """\
You are a compliance reporting assistant. Your task is to generate a final compliance report \
based on the results of clause compliance checks against a given set of guidelines. 

Analyze the provided compliance results and produce a structured report according to the specified schema. 
Ensure that if there are no noncompliant clauses, the report clearly indicates full compliance.
"""

COMPLIANCE_REPORT_USER_PROMPT = """\
A set of clauses within a contract were checked against GDPR compliance guidelines for the following vendor: {vendor_name}. 
The set of noncompliant clauses are given below.

Each section includes:
- **Clause:** The exact text of the contract clause.
- **Guideline:** The relevant GDPR guideline text.
- **Compliance Status:** Should be `False` for noncompliant clauses.
- **Notes:** Additional information or explanations.

{compliance_results}

Based on the above compliance results, generate a final compliance report following the `ComplianceReport` schema below. 
If there are no noncompliant clauses, the report should indicate that the contract is fully compliant.
"""


# Define five events to be used in a workflow
class ContractExtractionEvent(Event):
    contract_extraction: ContractExtraction


class MatchGuidelineEvent(Event):
    clause: ContractClause


class MatchGuidelineResultEvent(Event):
    result: ClauseComplianceCheck


class GenerateReportEvent(Event):
    match_results: List[ClauseComplianceCheck]


class LogEvent(Event):
    msg: str
    delta: bool = False


class ContractReviewWorkflow(Workflow):
    """Contract review workflow."""

    def __init__(
        self,
        parser: LlamaParse,
        guideline_retriever: BaseRetriever,
        llm: LLM | None = None,
        similarity_top_k: int = 20,
        output_dir: str = "data_out",
        **kwargs,
    ) -> None:
        """Init params."""
        super().__init__(**kwargs)

        self.parser = parser
        self.guideline_retriever = guideline_retriever
        
        self.llm = llm or OpenAI(model="gpt-4o-mini")
        self.similarity_top_k = similarity_top_k

        # if not exists, create
        out_path = Path(f"{curr_dir}/src_adw/{output_dir}/workflow_output")
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
            os.chmod(str(out_path), 0o0777)
        self.output_dir = out_path

    @step
    async def parse_contract(self, ctx: Context, ev: StartEvent) -> ContractExtractionEvent:
        # load output template file
        contract_extraction_path = Path(f"{self.output_dir}/contract_extraction.json")

        if contract_extraction_path.exists():
            if self._verbose:
                ctx.write_event_to_stream(LogEvent(msg="\n>> Loading contract from cache"))
            contract_extraction_dict = json.load(open(str(contract_extraction_path), "r"))
            contract_extraction = ContractExtraction.model_validate(contract_extraction_dict)
        else:
            if self._verbose:
                ctx.write_event_to_stream(LogEvent(msg="\n\n>> Reading contract"))

            # no need to parse contract, it's already in markdown
            # you can use LlamaParse to parse more complex PDFs + other docs

            docs = SimpleDirectoryReader(input_files=[ev.contract_path]).load_data()

            # extract from contract in the format of schema by prompt
            prompt = ChatPromptTemplate.from_messages([
                ("user", CONTRACT_EXTRACT_PROMPT)
            ])
            contract_extraction = await llm.astructured_predict(
                ContractExtraction,
                prompt,
                contract_data="\n".join([d.get_content(metadata_mode="all") for d in docs])
            )

            if not isinstance(contract_extraction, ContractExtraction):
                raise ValueError(f"Invalid extraction from contract: {contract_extraction}")
            
            # save output template to file
            # same extracted data in json format
            with open(contract_extraction_path, "w") as fp:
                fp.write(contract_extraction.model_dump_json())
        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=f"\n\n>> Contract data: {contract_extraction.model_dump()}"))

        return ContractExtractionEvent(contract_extraction=contract_extraction)

    @step
    async def dispatch_guideline_match(self, ctx: Context, ev: ContractExtractionEvent) -> MatchGuidelineEvent:
        """For each clause in the contract, find relevant guidelines.

        Use a map-reduce pattern. 
        
        """
        await ctx.set("num_clauses", len(ev.contract_extraction.clauses))
        await ctx.set("vendor_name", ev.contract_extraction.vendor_name)
        
        for clause in ev.contract_extraction.clauses:
            ctx.send_event(MatchGuidelineEvent(clause=clause, vendor_name=ev.contract_extraction.vendor_name))

    @step
    async def handle_guideline_match(self, ctx: Context, ev: MatchGuidelineEvent) -> MatchGuidelineResultEvent:
        """Handle matching clause against guideline."""

        # retrieve matching guideline
        query = f"""\
Please find the relevant guideline from {ev.vendor_name} that aligns with the following contract clause:

{ev.clause.clause_text}
"""
        guideline_docs = self.guideline_retriever.retrieve(query)
        guideline_text="\n\n".join([g.get_content() for g in guideline_docs])
        if self._verbose:
            ctx.write_event_to_stream(
                LogEvent(msg=f"\n\n>> Found guidelines: {guideline_text[:200]}...")
            )
        
        # extract from contract
        prompt = ChatPromptTemplate.from_messages([
            ("user", CONTRACT_MATCH_PROMPT)
        ])
        compliance_output = await llm.astructured_predict(
            ClauseComplianceCheck,
            prompt,
            clause_text=ev.clause.model_dump_json(),
            guideline_text=guideline_text
        )
        
        if not isinstance(compliance_output, ClauseComplianceCheck):
            raise ValueError(f"Invalid compliance check: {compliance_output}")

        return MatchGuidelineResultEvent(result=compliance_output)

    @step
    async def gather_guideline_match(self, ctx: Context, ev: MatchGuidelineResultEvent) -> GenerateReportEvent:
        """Handle matching clause against guideline."""
        num_clauses = await ctx.get("num_clauses")
        events = ctx.collect_events(ev, [MatchGuidelineResultEvent] * num_clauses)
        if events is None:
            return

        match_results = [e.result for e in events]
        # save match results
        match_results_path = Path(
            f"{self.output_dir}/match_results.jsonl"
        )
        with open(match_results_path, "w") as fp:
            for mr in match_results:
                fp.write(mr.model_dump_json() + "\n")
            
            
        return GenerateReportEvent(match_results=[e.result for e in events])

    @step
    async def generate_output(self, ctx: Context, ev: GenerateReportEvent) -> StopEvent:
        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg="\n\n>> Generating Compliance Report"))

        # if all clauses are compliant, return a compliant result
        non_compliant_results = [r for r in ev.match_results if not r.compliant]

        # generate compliance results string
        result_tmpl = """
1. **Clause**: {clause}
2. **Guideline:** {guideline}
3. **Compliance Status:** {compliance_status}
4. **Notes:** {notes}
"""
        non_compliant_strings = []
        for nr in non_compliant_results:
            non_compliant_strings.append(
                result_tmpl.format(
                    clause=nr.clause_text,
                    guideline=nr.matched_guideline.guideline_text,
                    compliance_status=nr.compliant,
                    notes=nr.notes
                )
            )
        non_compliant_str = "\n\n".join(non_compliant_strings)

        prompt = ChatPromptTemplate.from_messages([
            ("system", COMPLIANCE_REPORT_SYSTEM_PROMPT),
            ("user", COMPLIANCE_REPORT_USER_PROMPT)
        ])
        compliance_report = await llm.astructured_predict(
            ComplianceReport,
            prompt,
            compliance_results=non_compliant_str,
            vendor_name=await ctx.get("vendor_name")
        )

        return StopEvent(result={"report": compliance_report, "non_compliant_results": non_compliant_results})




# Create a workflow
workflow = ContractReviewWorkflow(
    parser=parser,
    guideline_retriever=retriever,
    llm=llm,
    verbose=True,
    timeout=None,  # don't worry about timeout to make sure it completes
)

# draw_all_possible_flows(ContractReviewWorkflow, filename="contract_workflow.html")



async def main():
    handler = workflow.run(contract_path=f"{curr_dir}/src_adw/data/vendor_agreement.md")
    async for event in handler.stream_events():
        if isinstance(event, LogEvent):
            if event.delta:
                print(event.msg, end="")
            else:
                print(event.msg)

    response_dict = await handler
    print(str(response_dict["report"]))
    print(response_dict["non_compliant_results"])




asyncio.run(main())