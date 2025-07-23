from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
from langchain_community.embeddings import DashScopeEmbeddings

from langchain_openai import ChatOpenAI

load_dotenv()

def create_llm(temperature=0, model_name=None, provider=None):
    """
    创建 LLM 实例的工具函数

    Args:
        temperature (float): 温度参数，控制输出的随机性，默认为 0
        model_name (str): 模型名称。如果为 None，则从 .env 文件读取
        max_tokens (int): 最大 token 数，默认为 4000
        provider (str): 模型提供商，支持 "openai", "qwen", "doubao"。如果为 None，则从 .env 文件读取

    Returns:
        ChatOpenAI: 配置好的 LLM 实例
    """
    # 加载环境变量
    if provider is None:
        provider = os.getenv('DEFAULT_SERVICE', 'QWEN')
        provider = provider.upper()

    # 如果 model_name 为 None，从环境变量读取
    if model_name is None:
        model_name = os.getenv('LLM_MODEL_NAME', 'gpt-4o')

    # 使用 match-case 实现 switch case 功能
    match provider.lower():
        case "openai":
            api_key = os.getenv('OPENAI_API_KEY')
            base_url = os.getenv('OPENAI_API_BASE', None)


        case "qwen":
            api_key = os.getenv('QWEN_API_KEY')
            model_name = os.getenv('QWEN_MODEL', 'qwen-max')
            base_url = os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')


        case "doubao":
            api_key = os.getenv('VOLC_API_KEY')
            model_name = os.getenv('VOLC_MODEL_NAME', 'doubao-1-5-lite-32k-250115')
            base_url = os.getenv('VOLC_BASE_URL', 'https://api.doubao.com/v1')

        case _:
            raise ValueError(f"不支持的提供商: {provider}。支持的提供商: openai, qwen, doubao")

    if not api_key:
        raise ValueError(f"请确保在 .env 文件中设置了 {provider}_API_KEY")

    # 使用统一的 ChatOpenAI 构造方法
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature
    )


# 预定义的响应模式
class SerpQueriesResponse(BaseModel):
    """SERP 查询响应模式"""
    queries: List[Dict[str, str]] = Field(..., description="搜索查询列表")


class LearningsResponse(BaseModel):
    """学习要点响应模式"""
    learnings: List[str] = Field(..., description="学习要点列表")
    summary: List[str] = Field(..., description="摘要列表")
    followUpQuestions: List[str] = Field(..., description="后续问题列表")


class ReportResponse(BaseModel):
    """报告响应模式"""
    reportMarkdown: str = Field(..., description="Markdown 格式的报告内容")


async def get_langchain_response(
    system_prompt: str,
    user_prompt: str,
    response_schema: Optional[BaseModel] = None,
    service_provider_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    使用 LangChain 获取 AI 响应

    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        response_schema: 响应模式（Pydantic 模型）
        service_provider_name: 服务提供商名称

    Returns:
        Dict[str, Any]: 解析后的响应
    """
    # 创建 LLM 实例
    llm = create_llm(provider=service_provider_name)

    # 构建消息
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    if response_schema:
        # 使用结构化输出
        structured_llm = llm.with_structured_output(response_schema)
        result = await structured_llm.ainvoke(messages)
        return result.dict()
    else:
        # 使用 JSON 输出解析器
        json_parser = JsonOutputParser()
        chain = llm | json_parser
        result = await chain.ainvoke(messages)
        return result


def create_embeddings(provider=None, model_name=None):
    """
    创建 Embeddings 实例的工具函数

    Args:
        provider (str): 模型提供商，支持 "openai", "qwen", "doubao"。如果为 None，则从 .env 文件读取
        model_name (str): 模型名称。如果为 None，则使用默认模型

    Returns:
        OpenAIEmbeddings: 配置好的 Embeddings 实例
    """
    # 加载环境变量

    # 如果 provider 为 None，从环境变量读取
    if provider is None:
        provider = os.getenv('DEFAULT_SERVICE', 'QWEN')
        provider = provider.upper()

    print(f"🔧 创建 Embeddings - 提供商: {provider}")

    # 使用 match-case 实现 switch case 功能
    match provider.lower():
        case "openai":
            api_key = os.getenv('OPENAI_API_KEY')
            base_url = os.getenv('OPENAI_API_BASE', None)
            if model_name is None:
                model_name = "text-embedding-3-small"

        case "qwen":
            api_key = os.getenv('QWEN_API_KEY')
            if model_name is None:
                model_name = "text-embedding-v1"
            base_url = os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
            # Qwen 需要特殊的配置
            print(f"🔧 Qwen 配置 - 使用兼容模式")

        # case "doubao":
        #     api_key = os.getenv('VOLC_API_KEY')
        #     if model_name is None:
        #         model_name = "doubao-embedding-vision-250615"
        #     base_url = os.getenv('VOLC_BASE_EMBEDDING_URL', 'https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal')

        case _:
            raise ValueError(f"不支持的提供商: {provider}。支持的提供商: openai, qwen")

    if not api_key:
        raise ValueError(f"请确保在 .env 文件中设置了 {provider}_API_KEY")

    print(f"🔧 Embeddings 配置 - 模型: {model_name}, Base URL: {base_url}")

    # 对于 Qwen，使用特殊的嵌入类
    if provider.lower() == "qwen":

        # embeddings = OpenAIEmbeddings(
        #     model=model_name,
        #     base_url=base_url,
        #     api_key=api_key
        # )
        embeddings = DashScopeEmbeddings(
            model=model_name,  # 根据实际情况选择模型版本
            dashscope_api_key=api_key  # 替换为你的通义千问API密钥
        )
    else:
        # 创建标准的 OpenAIEmbeddings 实例
        embeddings = create_embeddings()
    return embeddings


if __name__ == "__main__":
    llm = create_llm()
    print(llm.invoke("你好"))

    # 测试 embeddings
    embeddings = create_embeddings()
    print("Embeddings 模型创建成功")

    # Chroma 向量存储测试
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    print("=== Chroma 向量存储测试 ===")
    docs = [
        Document(page_content="Harry Potter is a wizard.", metadata={"source": "test1"}),
        Document(page_content="Hermione Granger is very smart.", metadata={"source": "test2"}),
        Document(page_content="Ron Weasley is Harry's friend.", metadata={"source": "test3"}),
    ]
    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="test_collection"
        )
        print("Chroma 向量存储创建成功！")
        # 简单检索
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        # 新写法（无警告）
        results = retriever.invoke("Who is Harry's friend?")
        print("检索结果：")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content} (source: {doc.metadata.get('source')})")
    except Exception as e:
        print(f"❌ Chroma 测试失败: {e}")
