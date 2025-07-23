from langchain_openai import ChatOpenAI

# --- Document Loading and Vector Store ---
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import DashScopeEmbeddings

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
from llm_utils import create_embeddings
from step1_init import book_quotes_list, chapter_summaries, create_llm, current_model_name, encode_book, encode_chapter_summaries, encode_quotes, hp_pdf_path

# --- Create or Load Vector Stores for Book Chunks, Chapter Summaries, and Book Quotes ---

# Check if the vector stores already exist on disk
if (
    os.path.exists("chunks_vector_store") and
    os.path.exists("chapter_summaries_vector_store") and
    os.path.exists("book_quotes_vectorstore")
):
    # If vector stores exist, load them using OpenAI embeddings
    embeddings = create_embeddings()
    chunks_vector_store = FAISS.load_local(
        "chunks_vector_store", embeddings, allow_dangerous_deserialization=True
    )
    chapter_summaries_vector_store = FAISS.load_local(
        "chapter_summaries_vector_store", embeddings, allow_dangerous_deserialization=True
    )
    book_quotes_vectorstore = FAISS.load_local(
        "book_quotes_vectorstore", embeddings, allow_dangerous_deserialization=True
    )
else:
    # If vector stores do not exist, encode and save them

    # 1. Encode the book into a vector store of chunks
    chunks_vector_store = encode_book(hp_pdf_path, chunk_size=1000, chunk_overlap=200)

    # 2. Encode the chapter summaries into a vector store
    chapter_summaries_vector_store = encode_chapter_summaries(chapter_summaries)

    # 3. Encode the book quotes into a vector store
    book_quotes_vectorstore = encode_quotes(book_quotes_list)

    # 4. Save the vector stores to disk for future use
    chunks_vector_store.save_local("chunks_vector_store")
    chapter_summaries_vector_store.save_local("chapter_summaries_vector_store")
    book_quotes_vectorstore.save_local("book_quotes_vectorstore")


# --- Create Query Retrievers from Vector Stores ---

# The following retrievers are used to fetch relevant documents from the vector stores
# based on a query. The number of results returned can be controlled via the 'k' parameter.

# Retriever for book chunks (returns the top 1 most relevant chunk)
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 1})

# Retriever for chapter summaries (returns the top 1 most relevant summary)
chapter_summaries_query_retriever = chapter_summaries_vector_store.as_retriever(search_kwargs={"k": 1})

# Retriever for book quotes (returns the top 10 most relevant quotes)
book_quotes_query_retriever = book_quotes_vectorstore.as_retriever(search_kwargs={"k": 10})


def retrieve_context_per_question(state):
    """
    Retrieves relevant context for a given question by aggregating content from:
    - Book chunks
    - Chapter summaries
    - Book quotes

    Args:
        state (dict): A dictionary containing the question to answer, with key "question".

    Returns:
        dict: A dictionary with keys:
            - "context": Aggregated context string from all sources.
            - "question": The original question.
    """
    question = state["question"]

    # Retrieve relevant book chunks
    print("Retrieving relevant chunks...")
    docs = chunks_query_retriever.get_relevant_documents(question)
    context = " ".join(doc.page_content for doc in docs)

    # Retrieve relevant chapter summaries
    print("Retrieving relevant chapter summaries...")
    docs_summaries = chapter_summaries_query_retriever.get_relevant_documents(question)
    context_summaries = " ".join(
        f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries
    )

    # Retrieve relevant book quotes
    print("Retrieving relevant book quotes...")
    docs_book_quotes = book_quotes_query_retriever.get_relevant_documents(question)
    book_quotes = " ".join(doc.page_content for doc in docs_book_quotes)

    # Aggregate all contexts and escape problematic characters
    all_contexts = context + context_summaries + book_quotes
    all_contexts = escape_quotes(all_contexts)

    return {"context": all_contexts, "question": question}


# --- LLM-based Function to Filter Only Relevant Retrieved Content ---

# Prompt template for filtering relevant content from retrieved documents
keep_only_relevant_content_prompt_template = """
You receive a query: {query} and retrieved documents: {retrieved_documents} from a vector store.
You need to filter out all the non relevant information that doesn't supply important information regarding the {query}.
Your goal is just to filter out the non relevant information.
You can remove parts of sentences that are not relevant to the query or remove whole sentences that are not relevant to the query.
DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
Output the filtered relevant content.
"""

# Output schema for the filtered relevant content
class KeepRelevantContent(BaseModel):
    relevant_content: str = Field(
        description="The relevant content from the retrieved documents that is relevant to the query."
    )

# Create the prompt for the LLM
keep_only_relevant_content_prompt = PromptTemplate(
    template=keep_only_relevant_content_prompt_template,
    input_variables=["query", "retrieved_documents"],
)

# Initialize the LLM for filtering relevant content
keep_only_relevant_content_llm = create_llm()

# Create the LLM chain for filtering relevant content
# KeepRelevantContent 是一个 Pydantic 数据模型，用于定义和约束 LLM 输出的结构。
# 它的作用是指定 LLM 过滤后返回的内容格式，确保输出中只包含与查询相关的内容（relevant_content 字段），
# 并且该内容是从检索到的文档中筛选出来的、与用户问题最相关的信息。
keep_only_relevant_content_chain = (
    keep_only_relevant_content_prompt
    | keep_only_relevant_content_llm.with_structured_output(KeepRelevantContent)
)

def keep_only_relevant_content(state):
    """
    Filters and keeps only the relevant content from the retrieved documents that is relevant to the query.

    Args:
        state (dict): A dictionary containing:
            - "question": The query question.
            - "context": The retrieved documents as a string.

    Returns:
        dict: A dictionary containing:
            - "relevant_context": The filtered relevant content.
            - "context": The original context.
            - "question": The original question.
    """
    question = state["question"]
    context = state["context"]

    input_data = {
        "query": question,
        "retrieved_documents": context
    }

    print("Keeping only the relevant content...")
    pprint("--------------------")
    output = keep_only_relevant_content_chain.invoke(input_data)
    relevant_content = output.relevant_content
    relevant_content = "".join(relevant_content)
    relevant_content = escape_quotes(relevant_content)

    return {
        "relevant_context": relevant_content,
        "context": context,
        "question": question
    }


    