from typing import TypedDict, List, Dict

from dataclasses import Field

# --- LangChain and LLM Imports ---
from langchain_openai import ChatOpenAI

# --- Document Loading and Vector Store ---
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

# --- Prompting and Document Utilities ---
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

# --- Core and Output Parsers ---
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.graph import MermaidDrawMethod

# --- LangGraph for Workflow Graphs ---
from langgraph.graph import END, StateGraph

# --- Standard Library Imports ---
from time import monotonic
from dotenv import load_dotenv
from pprint import pprint
import os

# --- Datasets and Typing ---
from datasets import Dataset
from typing_extensions import TypedDict
from IPython.display import display, Image
from typing import List, TypedDict

# --- RAGAS Metrics for Evaluation ---
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_similarity
)

import langgraph
from helper_functions import escape_quotes
from step2_retriver import book_quotes_query_retriever, chapter_summaries_query_retriever, chunks_query_retriever, keep_only_relevant_content, retrieve_context_per_question
from step3_question_answer import answer_question_from_context, rewrite_question
from step1_init import create_llm, current_model_name, groq_api_key

from langchain.embeddings import DashScopeEmbeddings

class PlanExecute(TypedDict):
    """
    Represents the state at each step of the plan execution pipeline.

    Attributes:
        curr_state (str): The current state or status of the execution.
        question (str): The original user question.
        anonymized_question (str): The anonymized version of the question (entities replaced with variables).
        query_to_retrieve_or_answer (str): The query to be used for retrieval or answering.
        plan (List[str]): The current plan as a list of steps to execute.
        past_steps (List[str]): List of steps that have already been executed.
        mapping (dict): Mapping of anonymized variables to original named entities.
        curr_context (str): The current context used for answering or retrieval.
        aggregated_context (str): The accumulated context from previous steps.
        tool (str): The tool or method used for the current step (e.g., retrieval, answer).
        response (str): The response or output generated at this step.
    """
    curr_state: str
    question: str
    anonymized_question: str
    query_to_retrieve_or_answer: str
    plan: List[str]
    past_steps: List[str]
    mapping: Dict[str, str]
    curr_context: str
    aggregated_context: str
    tool: str
    response: str



from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# -----------------------------------------------
# Planning Component for Multi-Step Question Answering
# -----------------------------------------------


# Define the output schema for the plan
class Plan(BaseModel):
    """
    Represents a step-by-step plan to answer a given question.
    Attributes:
        steps (List[str]): Ordered list of steps to follow.
    """
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

# Prompt template for generating a plan from a question
planner_prompt = """
For the given query {question}, come up with a simple step by step plan of how to figure out the answer.

This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
"""

planner_prompt = PromptTemplate(
    template=planner_prompt,
    input_variables=["question"],
)

# Initialize the LLM for planning (using GPT-4o)
planner_llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o",
    max_tokens=2000
)

# Compose the planning chain: prompt -> LLM -> structured output
planner = planner_prompt | planner_llm.with_structured_output(Plan)


# -----------------------------------------------------------
# Chain to Refine a Plan into Executable Steps for Retrieval/Answering
# -----------------------------------------------------------

# Prompt template for refining a plan so that each step is executable by a retrieval or answer operation
break_down_plan_prompt_template = """
You receive a plan {plan} which contains a series of steps to follow in order to answer a query.
You need to go through the plan and refine it according to these rules:
1. Every step must be executable by one of the following:
    i. Retrieving relevant information from a vector store of book chunks
    ii. Retrieving relevant information from a vector store of chapter summaries
    iii. Retrieving relevant information from a vector store of book quotes
    iv. Answering a question from a given context.
2. Every step should contain all the information needed to execute it.

Output the refined plan.
"""

# Create a PromptTemplate for the LLM
break_down_plan_prompt = PromptTemplate(
    template=break_down_plan_prompt_template,
    input_variables=["plan"],
)

# Initialize the LLM for plan breakdown (using GPT-4o)
break_down_plan_llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o",
    max_tokens=2000
)

# Compose the chain: prompt -> LLM -> structured output (Plan)
break_down_plan_chain = break_down_plan_prompt | break_down_plan_llm.with_structured_output(Plan)


# -----------------------------------------------------------
# Replanner: Update Plan Based on Progress and Aggregated Context
# -----------------------------------------------------------

# Define a Pydantic model for the possible results of the replanning action
class ActPossibleResults(BaseModel):
    """
    Represents the possible results of the replanning action.

    Attributes:
        plan (Plan): The updated plan to follow in the future.
        explanation (str): Explanation of the action taken or the reasoning behind the plan update.
    """
    plan: Plan = Field(description="Plan to follow in future.")
    explanation: str = Field(description="Explanation of the action.")

# Create a JSON output parser for the ActPossibleResults schema
act_possible_results_parser = JsonOutputParser(pydantic_object=ActPossibleResults)

# Prompt template for replanning, instructing the LLM to update the plan based on the current state
replanner_prompt_template = """
For the given objective, come up with a simple step by step plan of how to figure out the answer.
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Assume that the answer was not found yet and you need to update the plan accordingly, so the plan should never be empty.

Your objective was this:
{question}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

You already have the following context:
{aggregated_context}

Update your plan accordingly. If further steps are needed, fill out the plan with only those steps.
Do not return previously done steps as part of the plan.

The format is JSON so escape quotes and new lines.

{format_instructions}
"""

# Create a PromptTemplate object for the replanner
replanner_prompt = PromptTemplate(
    template=replanner_prompt_template,
    input_variables=["question", "plan", "past_steps", "aggregated_context"],
    partial_variables={"format_instructions": act_possible_results_parser.get_format_instructions()},
)

# Initialize the LLM for replanning (using GPT-4o)
replanner_llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o",
    max_tokens=2000
)

# Compose the replanner chain: prompt -> LLM -> output parser
replanner = replanner_prompt | replanner_llm | act_possible_results_parser


# -----------------------------------------------------------
# Task Handler: Decide Which Tool to Use for Each Task
# -----------------------------------------------------------

# Prompt template for the task handler LLM
tasks_handler_prompt_template = """
You are a task handler that receives a task {curr_task} and have to decide with tool to use to execute the task.
You have the following tools at your disposal:
Tool A: a tool that retrieves relevant information from a vector store of book chunks based on a given query.
- use Tool A when you think the current task should search for information in the book chunks.
Tool B: a tool that retrieves relevant information from a vector store of chapter summaries based on a given query.
- use Tool B when you think the current task should search for information in the chapter summaries.
Tool C: a tool that retrieves relevant information from a vector store of quotes from the book based on a given query.
- use Tool C when you think the current task should search for information in the book quotes.
Tool D: a tool that answers a question from a given context.
- use Tool D ONLY when you the current task can be answered by the aggregated context {aggregated_context}

You also receive the last tool used {last_tool}
if {last_tool} was retrieve_chunks, use other tools than Tool A.

You also have the past steps {past_steps} that you can use to make decisions and understand the context of the task.
You also have the initial user's question {question} that you can use to make decisions and understand the context of the task.
if you decide to use Tools A,B or C, output the query to be used for the tool and also output the relevant tool.
if you decide to use Tool D, output the question to be used for the tool, the context, and also that the tool to be used is Tool D.
"""

# Output schema for the task handler
class TaskHandlerOutput(BaseModel):
    """
    Output schema for the task handler.
    - query: The query to be either retrieved from the vector store, or the question that should be answered from context.
    - curr_context: The context to be based on in order to answer the query.
    - tool: The tool to be used; should be one of 'retrieve_chunks', 'retrieve_summaries', 'retrieve_quotes', or 'answer_from_context'.
    """
    query: str = Field(description="The query to be either retrieved from the vector store, or the question that should be answered from context.")
    curr_context: str = Field(description="The context to be based on in order to answer the query.")
    tool: str = Field(description="The tool to be used should be either retrieve_chunks, retrieve_summaries, retrieve_quotes, or answer_from_context.")

# Create the prompt object for the task handler
task_handler_prompt = PromptTemplate(
    template=tasks_handler_prompt_template,
    input_variables=["curr_task", "aggregated_context", "last_tool", "past_steps", "question"],
)

# Initialize the LLM for the task handler (using GPT-4o)
task_handler_llm = create_llm(max_token=2000)

# Compose the task handler chain: prompt -> LLM -> structured output
task_handler_chain = task_handler_prompt | task_handler_llm.with_structured_output(TaskHandlerOutput)


# -----------------------------------------------------------
# Anonymize Question Chain: Replace Named Entities with Variables
# -----------------------------------------------------------

# Define a Pydantic model for the anonymized question output
# AnonymizeQuestion 的作用是：将输入问题中的所有命名实体（如人名、地名、组织名等）用变量（如 X、Y、Z 等）替换，
# 并记录原始命名实体与变量之间的映射关系。
# 这样可以让后续处理流程在不暴露具体实体的情况下进行推理和处理，同时保留还原原始问题的能力。该类还会输出一个解释说明本次匿名化的过程。

class AnonymizeQuestion(BaseModel):
  """
  Output schema for the anonymized question.
  Attributes:
    anonymized_question (str): The question with named entities replaced by variables.
    mapping (dict): Mapping of variables to original named entities.
    explanation (str): Explanation of the anonymization process.
  """
  anonymized_question: str = Field(description="Anonymized question.")
  mapping: dict = Field(description="Mapping of original name entities to variables.")
  explanation: str = Field(description="Explanation of the action.")

# Create a JSON output parser for the AnonymizeQuestion schema
anonymize_question_parser = JsonOutputParser(pydantic_object=AnonymizeQuestion)

# Prompt template for the LLM to anonymize questions
anonymize_question_prompt_template = """
You are a question anonymizer. The input you receive is a string containing several words that
construct a question {question}. Your goal is to change all name entities in the input to variables, and remember the mapping of the original name entities to the variables.
Example 1:
  if the input is "who is harry potter?" the output should be "who is X?" and the mapping should be {{"X": "harry potter"}}
Example 2:
  if the input is "how did the bad guy played with the alex and rony?"
  the output should be "how did the X played with the Y and Z?" and the mapping should be {{"X": "bad guy", "Y": "alex", "Z": "rony"}}
You must replace all name entities in the input with variables, and remember the mapping of the original name entities to the variables.
Output the anonymized question and the mapping in a JSON format.
{format_instructions}
"""

# Create the PromptTemplate object for the anonymization task
anonymize_question_prompt = PromptTemplate(
  template=anonymize_question_prompt_template,
  input_variables=["question"],
  partial_variables={"format_instructions": anonymize_question_parser.get_format_instructions()},
)

# Initialize the LLM for anonymization (using GPT-4o)
anonymize_question_llm = ChatOpenAI(
  temperature=0,
  model_name="gpt-4o",
  max_tokens=2000
)

# Compose the anonymization chain: prompt -> LLM -> output parser
anonymize_question_chain = (
  anonymize_question_prompt
  | anonymize_question_llm
  | anonymize_question_parser
)



from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# -----------------------------------------------------------
# De-Anonymize Plan Chain: Replace Variables in Plan with Mapped Words
# -----------------------------------------------------------


class DeAnonymizePlan(BaseModel):
    """
    Output schema for the de-anonymized plan.
    Attributes:
        plan (List): Plan to follow in future, with all variables replaced by the mapped words.
    """
    plan: List = Field(
        description="Plan to follow in future. with all the variables replaced with the mapped words."
    )

# Prompt template for de-anonymizing a plan
de_anonymize_plan_prompt_template = (
    "You receive a list of tasks: {plan}, where some of the words are replaced with mapped variables. "
    "You also receive the mapping for those variables to words {mapping}. "
    "Replace all the variables in the list of tasks with the mapped words. "
    "If no variables are present, return the original list of tasks. "
    "In any case, just output the updated list of tasks in a JSON format as described here, "
    "without any additional text apart from the JSON."
)

# Create the PromptTemplate object for the de-anonymization task
de_anonymize_plan_prompt = PromptTemplate(
    template=de_anonymize_plan_prompt_template,
    input_variables=["plan", "mapping"],
)

# Initialize the LLM for de-anonymization (using GPT-4o)
de_anonymize_plan_llm = create_llm(2000)

# Compose the de-anonymization chain: prompt -> LLM -> structured output
de_anonymize_plan_chain = (
    de_anonymize_plan_prompt
    | de_anonymize_plan_llm.with_structured_output(DeAnonymizePlan)
)




# -----------------------------------------------------------
# LLM Chain: Check if a Question Can Be Fully Answered from Context
# -----------------------------------------------------------

# Define the output schema for the LLM's response
# CanBeAnsweredAlready 是一个用于判断“问题是否可以完全由给定上下文回答”的输出数据结构（Pydantic模型）。
# 具体来说，它有一个布尔类型的字段 can_be_answered，表示基于上下文能否完整回答该问题。

class CanBeAnsweredAlready(BaseModel):
    """
    Output schema for checking if the question can be fully answered from the given context.
    Attributes:
        can_be_answered (bool): Whether the question can be fully answered or not based on the given context.
    """
    can_be_answered: bool = Field(
        description="Whether the question can be fully answered or not based on the given context."
    )

# Prompt template for the LLM to determine answerability
can_be_answered_already_prompt_template = """
You receive a query: {question} and a context: {context}.
You need to determine if the question can be fully answered relying only on the given context.
The only information you have and can rely on is the context you received.
You have no prior knowledge of the question or the context.
If you think the question can be answered based on the context, output 'true', otherwise output 'false'.
"""

# Create the PromptTemplate object
can_be_answered_already_prompt = PromptTemplate(
    template=can_be_answered_already_prompt_template,
    input_variables=["question", "context"],
)

# Initialize the LLM for this task (using GPT-4o)
can_be_answered_already_llm = create_llm(2000)

# Compose the chain: prompt -> LLM -> structured output
can_be_answered_already_chain = (
    can_be_answered_already_prompt
    | can_be_answered_already_llm.with_structured_output(CanBeAnsweredAlready)
)


from pprint import pprint

def run_task_handler_chain(state: PlanExecute):
    """
    Run the task handler chain to decide which tool to use to execute the task.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state of the plan execution.
    """
    state["curr_state"] = "task_handler"
    print("the current plan is:")
    print(state["plan"])
    pprint("--------------------")

    # Initialize past_steps if not present
    if not state['past_steps']:
        state["past_steps"] = []

    # Get the current task from the plan
    curr_task = state["plan"][0]

    # Prepare inputs for the task handler chain
    inputs = {
        "curr_task": curr_task,
        "aggregated_context": state["aggregated_context"],
        "last_tool": state["tool"],
        "past_steps": state["past_steps"],
        "question": state["question"]
    }

    # Invoke the task handler chain
    output = task_handler_chain.invoke(inputs)

    # Update state with the completed task
    state["past_steps"].append(curr_task)
    state["plan"].pop(0)

    # Decide which tool to use based on output
    if output.tool == "retrieve_chunks":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"] = "retrieve_chunks"
    elif output.tool == "retrieve_summaries":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"] = "retrieve_summaries"
    elif output.tool == "retrieve_quotes":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"] = "retrieve_quotes"
    elif output.tool == "answer_from_context":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"] = "answer"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")
    return state


def retrieve_or_answer(state: PlanExecute):
    """
    Decide whether to retrieve or answer the question based on the current state.

    Args:
        state: The current state of the plan execution.

    Returns:
        String indicating the chosen tool.
    """
    state["curr_state"] = "decide_tool"
    print("deciding whether to retrieve or answer")
    if state["tool"] == "retrieve_chunks":
        return "chosen_tool_is_retrieve_chunks"
    elif state["tool"] == "retrieve_summaries":
        return "chosen_tool_is_retrieve_summaries"
    elif state["tool"] == "retrieve_quotes":
        return "chosen_tool_is_retrieve_quotes"
    elif state["tool"] == "answer":
        return "chosen_tool_is_answer"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")


def run_qualitative_chunks_retrieval_workflow(state):
    """
    Run the qualitative chunks retrieval workflow.

    Args:
        state: The current state of the plan execution.

    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "retrieve_chunks"
    print("Running the qualitative chunks retrieval workflow...")
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    # Stream outputs from the workflow app
    for output in qualitative_chunks_retrieval_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass
        pprint("--------------------")
    # Aggregate the retrieved context
    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output['relevant_context']
    return state


def run_qualitative_summaries_retrieval_workflow(state):
    """
    Run the qualitative summaries retrieval workflow.

    Args:
        state: The current state of the plan execution.

    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "retrieve_summaries"
    print("Running the qualitative summaries retrieval workflow...")
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    for output in qualitative_summaries_retrieval_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass
        pprint("--------------------")
    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output['relevant_context']
    return state


def run_qualitative_book_quotes_retrieval_workflow(state):
    """
    Run the qualitative book quotes retrieval workflow.

    Args:
        state: The current state of the plan execution.

    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "retrieve_book_quotes"
    print("Running the qualitative book quotes retrieval workflow...")
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    for output in qualitative_book_quotes_retrieval_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass
        pprint("--------------------")
    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output['relevant_context']
    return state


def run_qualtative_answer_workflow(state):
    """
    Run the qualitative answer workflow.

    Args:
        state: The current state of the plan execution.

    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "answer"
    print("Running the qualitative answer workflow...")
    question = state["query_to_retrieve_or_answer"]
    context = state["curr_context"]
    inputs = {"question": question, "context": context}
    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass
        pprint("--------------------")
    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output["answer"]
    return state


def run_qualtative_answer_workflow_for_final_answer(state):
    """
    Run the qualitative answer workflow for the final answer.

    Args:
        state: The current state of the plan execution.

    Returns:
        The state with the updated response.
    """
    state["curr_state"] = "get_final_answer"
    print("Running the qualitative answer workflow for final answer...")
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, value in output.items():
            pass
        pprint("--------------------")
    state["response"] = value
    return state


def anonymize_queries(state: PlanExecute):
    """
    Anonymizes the question.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the anonymized question and mapping.
    """
    state["curr_state"] = "anonymize_question"
    print("Anonymizing question")
    pprint("--------------------")
    anonymized_question_output = anonymize_question_chain.invoke(state['question'])
    anonymized_question = anonymized_question_output["anonymized_question"]
    print(f'anonimized_querry: {anonymized_question}')
    pprint("--------------------")
    mapping = anonymized_question_output["mapping"]
    state["anonymized_question"] = anonymized_question
    state["mapping"] = mapping
    return state


def deanonymize_queries(state: PlanExecute):
    """
    De-anonymizes the plan.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the de-anonymized plan.
    """
    state["curr_state"] = "de_anonymize_plan"
    print("De-anonymizing plan")
    pprint("--------------------")
    deanonimzed_plan = de_anonymize_plan_chain.invoke({"plan": state["plan"], "mapping": state["mapping"]})
    state["plan"] = deanonimzed_plan.plan
    print(f'de-anonimized_plan: {deanonimzed_plan.plan}')
    return state


def plan_step(state: PlanExecute):
    """
    Plans the next step.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the plan.
    """
    state["curr_state"] = "planner"
    print("Planning step")
    pprint("--------------------")
    plan = planner.invoke({"question": state['anonymized_question']})
    state["plan"] = plan.steps
    print(f'plan: {state["plan"]}')
    return state


def break_down_plan_step(state: PlanExecute):
    """
    Breaks down the plan steps into retrievable or answerable tasks.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the refined plan.
    """
    state["curr_state"] = "break_down_plan"
    print("Breaking down plan steps into retrievable or answerable tasks")
    pprint("--------------------")
    refined_plan = break_down_plan_chain.invoke(state["plan"])
    state["plan"] = refined_plan.steps
    return state


def replan_step(state: PlanExecute):
    """
    Replans the next step.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the plan.
    """
    state["curr_state"] = "replan"
    print("Replanning step")
    pprint("--------------------")
    inputs = {
        "question": state["question"],
        "plan": state["plan"],
        "past_steps": state["past_steps"],
        "aggregated_context": state["aggregated_context"]
    }
    output = replanner.invoke(inputs)
    state["plan"] = output['plan']['steps']
    return state


def can_be_answered(state: PlanExecute):
    """
    Determines if the question can be answered.

    Args:
        state: The current state of the plan execution.

    Returns:
        String indicating whether the original question can be answered or not.
    """
    state["curr_state"] = "can_be_answered_already"
    print("Checking if the ORIGINAL QUESTION can be answered already")
    pprint("--------------------")
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    output = can_be_answered_already_chain.invoke(inputs)
    if output.can_be_answered == True:
        print("The ORIGINAL QUESTION can be fully answered already.")
        pprint("--------------------")
        print("the aggregated context is:")
        print(text_wrap(state["aggregated_context"]))
        print("--------------------")
        return "can_be_answered_already"
    else:
        print("The ORIGINAL QUESTION cannot be fully answered yet.")
        pprint("--------------------")
        return "cannot_be_answered_yet"




### 画图的
from langgraph.graph import StateGraph


# -----------------------------------------------------------
# Define the Plan-and-Execute Agent Workflow Graph
# -----------------------------------------------------------

# Initialize the workflow graph with the PlanExecute state
agent_workflow = StateGraph(PlanExecute)

# -------------------------
# Add Nodes (Steps/Functions)
# -------------------------

# 1. Anonymize the question (replace named entities with variables)
agent_workflow.add_node("anonymize_question", anonymize_queries)

# 2. Generate a step-by-step plan for the anonymized question
agent_workflow.add_node("planner", plan_step)

# 3. De-anonymize the plan (replace variables back with original entities)
agent_workflow.add_node("de_anonymize_plan", deanonymize_queries)

# 4. Break down the plan into retrievable/answerable tasks
agent_workflow.add_node("break_down_plan", break_down_plan_step)

# 5. Decide which tool to use for the current task
agent_workflow.add_node("task_handler", run_task_handler_chain)

# 6. Retrieve relevant book chunks
agent_workflow.add_node("retrieve_chunks", run_qualitative_chunks_retrieval_workflow)

# 7. Retrieve relevant chapter summaries
agent_workflow.add_node("retrieve_summaries", run_qualitative_summaries_retrieval_workflow)

# 8. Retrieve relevant book quotes
agent_workflow.add_node("retrieve_book_quotes", run_qualitative_book_quotes_retrieval_workflow)

# 9. Answer the question from the aggregated context
agent_workflow.add_node("answer", run_qualtative_answer_workflow)

# 10. Replan if needed (update plan based on progress/context)
agent_workflow.add_node("replan", replan_step)

# 11. Get the final answer from the aggregated context
agent_workflow.add_node("get_final_answer", run_qualtative_answer_workflow_for_final_answer)

# -------------------------
# Define Workflow Edges (Transitions)
# -------------------------

# Set the entry point of the workflow
agent_workflow.set_entry_point("anonymize_question")

# Anonymize -> Plan
agent_workflow.add_edge("anonymize_question", "planner")

# Plan -> De-anonymize
agent_workflow.add_edge("planner", "de_anonymize_plan")

# De-anonymize -> Break down plan
agent_workflow.add_edge("de_anonymize_plan", "break_down_plan")

# Break down plan -> Task handler
agent_workflow.add_edge("break_down_plan", "task_handler")

# Task handler -> (conditional) Retrieve or Answer
agent_workflow.add_conditional_edges(
    "task_handler",
    retrieve_or_answer,
    {
        "chosen_tool_is_retrieve_chunks": "retrieve_chunks",
        "chosen_tool_is_retrieve_summaries": "retrieve_summaries",
        "chosen_tool_is_retrieve_quotes": "retrieve_book_quotes",
        "chosen_tool_is_answer": "answer"
    }
)

# Retrieval/Answer nodes -> Replan
agent_workflow.add_edge("retrieve_chunks", "replan")
agent_workflow.add_edge("retrieve_summaries", "replan")
agent_workflow.add_edge("retrieve_book_quotes", "replan")
agent_workflow.add_edge("answer", "replan")

# Replan -> (conditional) Get final answer or continue
agent_workflow.add_conditional_edges(
    "replan",
    can_be_answered,
    {
        "can_be_answered_already": "get_final_answer",
        "cannot_be_answered_yet": "break_down_plan"
    }
)

# Get final answer -> End
agent_workflow.add_edge("get_final_answer", END)

# -------------------------
# Compile and Visualize the Workflow
# -------------------------

plan_and_execute_app = agent_workflow.compile()

# Display the workflow graph as a Mermaid diagram
display(Image(plan_and_execute_app.get_graph(xray=True).draw_mermaid_png()))




def execute_plan_and_print_steps(inputs, recursion_limit=45):
    """
    Executes the plan-and-execute agent workflow and prints each step.

    Args:
        inputs (dict): The initial input state for the plan-and-execute agent.
        recursion_limit (int): Maximum number of steps to prevent infinite loops.

    Returns:
        tuple: (response, final_state)
            response (str): The final answer or message if not found.
            final_state (dict): The final state after execution.
    """
    # Configuration for the workflow (limits recursion to avoid infinite loops)
    config = {"recursion_limit": recursion_limit}
    try:
        # Stream the outputs from the plan_and_execute_app workflow
        for plan_output in plan_and_execute_app.stream(inputs, config=config):
            # Iterate through each step's output and print the current state
            for _, agent_state_value in plan_output.items():
                pass  # agent_state_value holds the latest state after each node execution
                print(f' curr step: {agent_state_value}')
        # Extract the final response from the last state
        response = agent_state_value['response']
    except langgraph.pregel.GraphRecursionError:
        # Handle the case where the recursion limit is reached
        response = "The answer wasn't found in the data."
    # Save the final state for further inspection or evaluation
    final_state = agent_state_value
    # Print the final answer in a wrapped format for readability
    print(text_wrap(f' the final answer is: {response}'))
    return response, final_state