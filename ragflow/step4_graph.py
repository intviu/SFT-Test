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

# 这一堆都是用LLM 进行校验

# --- LLM-based Function to Determine Relevance of Retrieved Content ---

# Prompt template for checking if the retrieved context is relevant to the query


is_relevant_content_prompt_template = """
You receive a query: {query} and a context: {context} retrieved from a vector store.
You need to determine if the document is relevant to the query.
{format_instructions}
"""

# Output schema for the relevance check
class Relevance(BaseModel):
    is_relevant: bool = Field(description="Whether the document is relevant to the query.")
    explanation: str = Field(description="An explanation of why the document is relevant or not.")

# JSON parser for the output schema
is_relevant_json_parser = JsonOutputParser(pydantic_object=Relevance)

# Initialize the LLM for relevance checking
is_relevant_llm = create_llm()

# Create the prompt object for the LLM
is_relevant_content_prompt = PromptTemplate(
    template=is_relevant_content_prompt_template,
    input_variables=["query", "context"],
    partial_variables={"format_instructions": is_relevant_json_parser.get_format_instructions()},
)

# Combine prompt, LLM, and parser into a chain
is_relevant_content_chain = is_relevant_content_prompt | is_relevant_llm | is_relevant_json_parser

def is_relevant_content(state):
    """
    Determines if the retrieved context is relevant to the query.

    Args:
        state (dict): A dictionary containing:
            - "question": The query question.
            - "context": The retrieved context to check for relevance.

    Returns:
        str: "relevant" if the context is relevant, "not relevant" otherwise.
    """
    question = state["question"]
    context = state["context"]

    input_data = {
        "query": question,
        "context": context
    }

    # Invoke the LLM chain to determine if the document is relevant
    output = is_relevant_content_chain.invoke(input_data)
    print("Determining if the document is relevant...")
    if output["is_relevant"]:
        print("The document is relevant.")
        return "relevant"
    else:
        print("The document is not relevant.")
        return "not relevant"



# --- LLM Chain to Check if an Answer is Grounded in the Provided Context ---

# Define the output schema for the grounding check
class IsGroundedOnFacts(BaseModel):
    """
    Output schema for checking if the answer is grounded in the provided context.
    """
    grounded_on_facts: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

# Initialize the LLM for fact-checking (using GPT-4o)
is_grounded_on_facts_llm = create_llm()

# Define the prompt template for fact-checking
is_grounded_on_facts_prompt_template = """
You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
You don't mind if it doesn't make sense, as long as it is grounded in the context.
Output a JSON containing the answer to the question, and apart from the JSON format don't output any additional text.
"""

# Create the prompt object
is_grounded_on_facts_prompt = PromptTemplate(
    template=is_grounded_on_facts_prompt_template,
    input_variables=["context", "answer"],
)

# Create the LLM chain for fact-checking
is_grounded_on_facts_chain = (
    is_grounded_on_facts_prompt
    | is_grounded_on_facts_llm.with_structured_output(IsGroundedOnFacts)
)




# --- LLM Chain to Determine if a Question Can Be Fully Answered from Context ---
# Define the prompt template for the LLM
can_be_answered_prompt_template = """
You receive a query: {question} and a context: {context}.
You need to determine if the question can be fully answered based on the context.
{format_instructions}
"""

# Define the output schema for the LLM's response
class QuestionAnswer(BaseModel):
    can_be_answered: bool = Field(
        description="binary result of whether the question can be fully answered or not"
    )
    explanation: str = Field(
        description="An explanation of why the question can be fully answered or not."
    )

# Create a JSON parser for the output schema
can_be_answered_json_parser = JsonOutputParser(pydantic_object=QuestionAnswer)

# Create the prompt object for the LLM
answer_question_prompt = PromptTemplate(
    template=can_be_answered_prompt_template,
    input_variables=["question", "context"],
    partial_variables={"format_instructions": can_be_answered_json_parser.get_format_instructions()},
)

# Initialize the LLM (Groq Llama3) for this task
can_be_answered_llm = create_llm()

# Compose the chain: prompt -> LLM -> output parser
can_be_answered_chain = answer_question_prompt | can_be_answered_llm | can_be_answered_json_parser


def grade_generation_v_documents_and_question(state):
    """
    Grades the generated answer to a question based on:
    - Whether the answer is grounded in the provided context (fact-checking)
    - Whether the question can be fully answered from the context

    Args:
        state (dict): A dictionary containing:
            - "context": The context used to answer the question
            - "question": The original question
            - "answer": The generated answer

    Returns:
        str: One of "hallucination", "useful", or "not_useful"
    """

    # Extract relevant fields from state
    context = state["context"]
    answer = state["answer"]
    question = state["question"]

    # 1. Check if the answer is grounded in the provided context (fact-checking)
    print("Checking if the answer is grounded in the facts...")
    result = is_grounded_on_facts_chain.invoke({"context": context, "answer": answer})
    grounded_on_facts = result.grounded_on_facts

    if not grounded_on_facts:
        # If not grounded, label as hallucination
        print("The answer is hallucination.")
        return "hallucination"
    else:
        print("The answer is grounded in the facts.")

        # 2. Check if the question can be fully answered from the context
        input_data = {
            "question": question,
            "context": context
        }
        print("Determining if the question is fully answered...")
        output = can_be_answered_chain.invoke(input_data)
        can_be_answered = output["can_be_answered"]

        if can_be_answered:
            print("The question can be fully answered.")
            return "useful"
        else:
            print("The question cannot be fully answered.")
            return "not_useful"



### 测试一下

# --- Step-by-step Pipeline for Answering a Question with RAG and LLMs ---

# 1. Define the initial state with the question to answer
init_state = {"question": "who is fluffy?"}

# 2. Retrieve relevant context for the question from the vector stores (chunks, summaries, quotes)
context_state = retrieve_context_per_question(init_state)

# 3. Use an LLM to filter and keep only the content relevant to the question from the retrieved context
relevant_content_state = keep_only_relevant_content(context_state)

# 4. Check if the filtered content is relevant to the question using an LLM-based relevance check
is_relevant_content_state = is_relevant_content(relevant_content_state)

# 5. Use an LLM to answer the question based on the relevant context
answer_state = answer_question_from_context(relevant_content_state)

# 6. Grade the generated answer:
#    - Check if the answer is grounded in the provided context (fact-checking)
#    - Check if the question can be fully answered from the context
final_answer = grade_generation_v_documents_and_question(answer_state)

# 7. Print the final answer
print(answer_state["answer"])






# -----------------------------------------------
# Qualitative Retrieval Answer Graph Construction
# -----------------------------------------------

# Define the state for the workflow graph
# QualitativeRetievalAnswerGraphState 是一个用于定义工作流图（workflow graph）中状态（state）数据结构的类型注解（TypedDict）。
# 它指定了在“定性检索问答”流程中，每一步处理函数所接收和返回的状态字典必须包含哪些字段，以及这些字段的数据类型。
# 具体来说，QualitativeRetievalAnswerGraphState 继承自 TypedDict，包含如下字段：
# - question: str  # 问题文本
# - context: str   # 用于回答问题的上下文内容
# - answer: str    # 基于上下文生成的答案
# 这样定义后，整个工作流中的节点函数都可以明确知道输入和输出的状态结构，便于类型检查和代码规范。
class QualitativeRetievalAnswerGraphState(TypedDict):
    question: str
    context: str
    answer: str

# Create the workflow graph object
qualitative_retrieval_answer_workflow = StateGraph(QualitativeRetievalAnswerGraphState)

# -------------------------
# Define and Add Graph Nodes
# -------------------------
# Each node represents a function in the pipeline

# Node: Retrieve context for the question from vector stores
qualitative_retrieval_answer_workflow.add_node(
    "retrieve_context_per_question", retrieve_context_per_question
)

# Node: Use LLM to keep only relevant content from the retrieved context
qualitative_retrieval_answer_workflow.add_node(
    "keep_only_relevant_content", keep_only_relevant_content
)

# Node: Rewrite the question for better retrieval if needed
qualitative_retrieval_answer_workflow.add_node(
    "rewrite_question", rewrite_question
)

# Node: Answer the question from the relevant context using LLM
qualitative_retrieval_answer_workflow.add_node(
    "answer_question_from_context", answer_question_from_context
)

# -------------------------
# Build the Workflow Edges
# -------------------------

# Set the entry point of the workflow
qualitative_retrieval_answer_workflow.set_entry_point("retrieve_context_per_question")

# Edge: After retrieving context, filter to keep only relevant content
qualitative_retrieval_answer_workflow.add_edge(
    "retrieve_context_per_question", "keep_only_relevant_content"
)

# Conditional Edge: After filtering, check if content is relevant
# If relevant, answer the question; if not, rewrite the question
qualitative_retrieval_answer_workflow.add_conditional_edges(
    "keep_only_relevant_content",
    is_relevant_content,
    {
        "relevant": "answer_question_from_context",
        "not relevant": "rewrite_question"
    },
)

# Edge: After rewriting the question, retrieve context again
qualitative_retrieval_answer_workflow.add_edge(
    "rewrite_question", "retrieve_context_per_question"
)

# Conditional Edge: After answering, grade the answer
# If hallucination, try answering again; if not useful, rewrite question; if useful, end
qualitative_retrieval_answer_workflow.add_conditional_edges(
    "answer_question_from_context",
    grade_generation_v_documents_and_question,
    {
        "hallucination": "answer_question_from_context",
        "not_useful": "rewrite_question",
        "useful": END
    },
)

# Compile the workflow graph into an executable app
qualitative_retrieval_answer_retrival_app = qualitative_retrieval_answer_workflow.compile()

# Display the workflow graph as a Mermaid diagram
display(
    Image(
        qualitative_retrieval_answer_retrival_app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)


# --- Function: Check if Distilled Content is Grounded in the Original Context ---

# Prompt template for the LLM to determine grounding
is_distilled_content_grounded_on_content_prompt_template = """
You receive some distilled content: {distilled_content} and the original context: {original_context}.
You need to determine if the distilled content is grounded on the original context.
If the distilled content is grounded on the original context, set the grounded field to true.
If the distilled content is not grounded on the original context, set the grounded field to false.
{format_instructions}
"""

# Output schema for the LLM's response
class IsDistilledContentGroundedOnContent(BaseModel):
    grounded: bool = Field(
        description="Whether the distilled content is grounded on the original context."
    )
    explanation: str = Field(
        description="An explanation of why the distilled content is or is not grounded on the original context."
    )

# JSON parser for the output schema
is_distilled_content_grounded_on_content_json_parser = JsonOutputParser(
    pydantic_object=IsDistilledContentGroundedOnContent
)

# Create the prompt object for the LLM
is_distilled_content_grounded_on_content_prompt = PromptTemplate(
    template=is_distilled_content_grounded_on_content_prompt_template,
    input_variables=["distilled_content", "original_context"],
    partial_variables={
        "format_instructions": is_distilled_content_grounded_on_content_json_parser.get_format_instructions()
    },
)

# Initialize the LLM for this task
is_distilled_content_grounded_on_content_llm = create_llm()

# Compose the chain: prompt -> LLM -> output parser
is_distilled_content_grounded_on_content_chain = (
    is_distilled_content_grounded_on_content_prompt
    | is_distilled_content_grounded_on_content_llm
    | is_distilled_content_grounded_on_content_json_parser
)

def is_distilled_content_grounded_on_content(state):
    """
    Determines if the distilled content is grounded on the original context.

    Args:
        state (dict): A dictionary containing:
            - "relevant_context": The distilled content.
            - "context": The original context.

    Returns:
        str: "grounded on the original context" if grounded, otherwise "not grounded on the original context".
    """
    pprint("--------------------")
    print("Determining if the distilled content is grounded on the original context...")

    distilled_content = state["relevant_context"]
    original_context = state["context"]

    input_data = {
        "distilled_content": distilled_content,
        "original_context": original_context
    }

    # Invoke the LLM chain to check grounding
    output = is_distilled_content_grounded_on_content_chain.invoke(input_data)
    grounded = output["grounded"]

    if grounded:
        print("The distilled content is grounded on the original context.")
        return "grounded on the original context"
    else:
        print("The distilled content is not grounded on the original context.")
        return "not grounded on the original context"



# -----------------------------------------------------------
# Retrieval Functions for Different Context Types
# -----------------------------------------------------------

def retrieve_chunks_context_per_question(state):
    """
    Retrieves relevant context for a given question from the book chunks.

    Args:
        state (dict): A dictionary containing the question to answer, with key "question".

    Returns:
        dict: A dictionary with keys:
            - "context": Aggregated context string from relevant book chunks.
            - "question": The original question.
    """
    print("Retrieving relevant chunks...")
    question = state["question"]
    # Retrieve relevant book chunks using the retriever
    docs = chunks_query_retriever.get_relevant_documents(question)
    # Concatenate the content of the retrieved documents
    context = " ".join(doc.page_content for doc in docs)
    context = escape_quotes(context)
    return {"context": context, "question": question}

def retrieve_summaries_context_per_question(state):
    """
    Retrieves relevant context for a given question from chapter summaries.

    Args:
        state (dict): A dictionary containing the question to answer, with key "question".

    Returns:
        dict: A dictionary with keys:
            - "context": Aggregated context string from relevant chapter summaries.
            - "question": The original question.
    """
    print("Retrieving relevant chapter summaries...")
    question = state["question"]
    # Retrieve relevant chapter summaries using the retriever
    docs_summaries = chapter_summaries_query_retriever.get_relevant_documents(question)
    # Concatenate the content of the retrieved summaries, including chapter citation
    context_summaries = " ".join(
        f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries
    )
    context_summaries = escape_quotes(context_summaries)
    return {"context": context_summaries, "question": question}

def retrieve_book_quotes_context_per_question(state):
    """
    Retrieves relevant context for a given question from book quotes.

    Args:
        state (dict): A dictionary containing the question to answer, with key "question".

    Returns:
        dict: A dictionary with keys:
            - "context": Aggregated context string from relevant book quotes.
            - "question": The original question.
    """
    question = state["question"]
    print("Retrieving relevant book quotes...")
    # Retrieve relevant book quotes using the retriever
    docs_book_quotes = book_quotes_query_retriever.get_relevant_documents(question)
    # Concatenate the content of the retrieved quotes
    book_quotes = " ".join(doc.page_content for doc in docs_book_quotes)
    book_quotes_context = escape_quotes(book_quotes)
    return {"context": book_quotes_context, "question": question}


