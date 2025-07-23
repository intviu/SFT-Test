# --- LangChain and LLM Imports ---
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

from langchain.embeddings import DashScopeEmbeddings

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

# --- Helper Functions ---
from helper_functions import (
    num_tokens_from_string,
    replace_t_with_space,
    replace_double_lines_with_one_line,
    split_into_chapters,
    analyse_metric_results,
    escape_quotes,
    text_wrap,
    extract_book_quotes_as_documents
)

from langchain_community.embeddings import DashScopeEmbeddings
from llm_utils import create_embeddings, create_llm

# --- Load environment variables (e.g., API keys) ---
load_dotenv(override=True)

# --- Set environment variable for debugging (optional) ---
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "100000"


# Set the OpenAI API key from environment variable (for use by OpenAI LLMs)
os.environ["QWEN_API_KEY"] = os.getenv("QWEN_API_KEY")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Retrieve the Groq API key from environment variable (for use by Groq LLMs)
groq_api_key =''

# Define the path to the Harry Potter PDF file.
# This variable will be used throughout the notebook for loading and processing the book.
hp_pdf_path = "Harry_Potter.pdf"

# --- Split the PDF into chapters and preprocess the text ---

# 1. Split the PDF into chapters using the provided helper function.
#    This function takes the path to the PDF and returns a list of Document objects, each representing a chapter.
chapters = split_into_chapters(hp_pdf_path)

# 2. Clean up the text in each chapter by replacing unwanted characters (e.g., '\t') with spaces.
#    This ensures the text is consistent and easier to process downstream.
chapters = replace_t_with_space(chapters)

# 3. Print the number of chapters extracted to verify the result.
print(len(chapters))


# --- Load and Preprocess the PDF, then Extract Quotes ---

# 1. Load the PDF using PyPDFLoader
loader = PyPDFLoader(hp_pdf_path)
document = loader.load()

# 2. Clean the loaded document by replacing unwanted characters (e.g., '\t') with spaces
document_cleaned = replace_t_with_space(document)

# 3. Extract a list of quotes from the cleaned document as Document objects
book_quotes_list = extract_book_quotes_as_documents(document_cleaned)

# --- Summarization Prompt Template for LLM-based Summarization ---

# Define the template string for summarization.
# This template instructs the language model to write an extensive summary of the provided text.
summarization_prompt_template = """Write an extensive summary of the following:

{text}

SUMMARY:"""

# Create a PromptTemplate object using the template string.
# The input variable "text" will be replaced with the content to summarize.
summarization_prompt = PromptTemplate(
    template=summarization_prompt_template,
    input_variables=["text"]
)

current_model_name = "qwen-max"
# 让LLM写详细的摘要
def create_chapter_summary(chapter):
    """
    Creates a summary of a chapter using a large language model (LLM).

    Args:
        chapter: A Document object representing the chapter to summarize.

    Returns:
        A Document object containing the summary of the chapter.
    """

    # Extract the text content from the chapter
    chapter_txt = chapter.page_content

    # Specify the LLM model and configuration
    llm = create_llm()  
    gpt_35_turbo_max_tokens = 16000  # Maximum token limit for the model
    verbose = False  # Set to True for more detailed output

    # Calculate the number of tokens in the chapter text
    num_tokens = len(chapter_txt)

    # Choose the summarization chain type based on token count
    if num_tokens < gpt_35_turbo_max_tokens:
        # For shorter chapters, use the "stuff" chain type
        # 解释：load_summarize_chain 是一个用于构建文档摘要流程的函数，通常来自 langchain 库。它根据指定的 chain_type（如 "stuff" 或 "map_reduce"）自动选择合适的摘要处理方式。
        # 主要参数说明如下：
        # - llm：指定用于生成摘要的语言模型（如 ChatOpenAI 实例）。
        # - chain_type：摘要链类型。"stuff" 适用于较短文本，直接将全部内容输入模型；"map_reduce" 适用于长文本，先分段摘要再合并。
        # - prompt：用于指导模型生成摘要的提示词模板（PromptTemplate）。
        # - map_prompt/combine_prompt：在 "map_reduce" 模式下，分别用于分段摘要和合并摘要的提示词模板。
        # - verbose：是否输出详细的执行信息，便于调试。
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=summarization_prompt,
            verbose=verbose
        )
    else:
        # For longer chapters, use the "map_reduce" chain type
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=summarization_prompt,
            combine_prompt=summarization_prompt,
            verbose=verbose
        )

    # Start timer to measure summarization time
    start_time = monotonic()

    # Create a Document object for the chapter
    doc_chapter = Document(page_content=chapter_txt)

    # Generate the summary using the selected chain
    summary_result = chain.invoke([doc_chapter])

    # Print chain type and execution time for reference
    # Print chain type and execution time for reference
    print(f"Chain type: {chain.__class__.__name__}")
    print(f"Run time: {monotonic() - start_time}")

    # Clean up the summary text (remove double newlines, etc.)
    summary_text = replace_double_lines_with_one_line(summary_result["output_text"])
    print(summary_text)

    # Create a Document object for the summary, preserving chapter metadata
    doc_summary = Document(page_content=summary_text, metadata=chapter.metadata)

    return doc_summary


    # --- Generate Summaries for Each Chapter ---

# Initialize an empty list to store the summaries of each chapter
chapter_summaries = []

# Iterate over each chapter in the chapters list
# for chapter in chapters:
    # Generate a summary for the current chapter using the create_chapter_summary function
    # summary = create_chapter_summary(chapter)
    # Append the summary to the chapter_summaries list
    # chapter_summaries.append(summary)



def encode_book(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a FAISS vector store using OpenAI embeddings.

    Args:
        path (str): The path to the PDF file.
        chunk_size (int): The desired size of each text chunk.
        chunk_overlap (int): The amount of overlap between consecutive chunks.

    Returns:
        FAISS: A FAISS vector store containing the encoded book content.
    """

    # 1. Load the PDF document using PyPDFLoader
    loader = PyPDFLoader(path)
    documents = loader.load()

    # 2. Split the document into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # 3. Clean up the text chunks (replace unwanted characters)
    cleaned_texts = replace_t_with_space(texts)

    # 4. Create OpenAI embeddings and encode the cleaned text chunks into a FAISS vector store
    embeddings = create_embeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    # 5. Return the vector store
    return vectorstore


def encode_chapter_summaries(chapter_summaries):
    """
    Encodes a list of chapter summaries into a FAISS vector store using OpenAI embeddings.

    Args:
        chapter_summaries (list): A list of Document objects representing the chapter summaries.

    Returns:
        FAISS: A FAISS vector store containing the encoded chapter summaries.
    """
    # Create OpenAI embeddings instance
    embeddings = create_embeddings()

    # Encode the chapter summaries into a FAISS vector store
    chapter_summaries_vectorstore = FAISS.from_documents(chapter_summaries, embeddings)

    # Return the vector store
    return chapter_summaries_vectorstore


# 下面是对 encode_quotes 函数的解释：
# encode_quotes 函数的作用是：将一本书中提取出来的所有引用（quotes，以 Document 对象列表形式传入）通过 OpenAI 的 Embeddings 进行向量化编码，
# 并存入 FAISS 向量数据库，便于后续高效地进行相似性检索。
# 具体流程如下：
# 1. 创建 DashScopeEmbeddings 实例，用于将文本转换为向量。
# 2. 调用 FAISS.from_documents 方法，将所有 quote 文本（Document 列表）批量编码为向量，并存入 FAISS 向量库。
# 3. 返回这个包含所有 quote 向量的 FAISS 向量库对象。
def encode_quotes(book_quotes_list):
    """
    Encodes a list of book quotes into a FAISS vector store using OpenAI embeddings.

    Args:
        book_quotes_list (list): A list of Document objects, each representing a quote from the book.

    Returns:
        FAISS: A FAISS vector store containing the encoded book quotes.
    """
    # Create OpenAI embeddings instance
    embeddings = create_embeddings()

    # Encode the book quotes into a FAISS vector store
    quotes_vectorstore = FAISS.from_documents(book_quotes_list, embeddings)
    # 打印 quotes_vectorstore 的相关信息
    print("quotes_vectorstore 信息：")
    print("向量数量：", quotes_vectorstore.index.ntotal)
    print("是否有持久化路径：", getattr(quotes_vectorstore, "persist_path", None))
    print("文档数量：", len(quotes_vectorstore.docstore._dict) if hasattr(quotes_vectorstore, "docstore") else "未知")
    # Return the vector store
    return quotes_vectorstore
