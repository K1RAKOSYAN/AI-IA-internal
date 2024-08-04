from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate


def chunkEmbeddingAndUpsert():
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_SECRET_KEY")

    index_name = "ai-ia"
    headers_to_split_on = [("##", "Header 2")]

    fp = open("data.md", encoding="utf-8")

    data = fp.read()

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    md_header_splits = markdown_splitter.split_text(data)
    model_name = "text-embedding-3-large"
    embeddings = OpenAIEmbeddings(model=model_name, api_key=OPENAI_API_KEY)

    vector_store = PineconeVectorStore.from_documents(
        md_header_splits, embeddings, index_name=index_name
    )
    return vector_store


async def queryEmbedding(prompt="What is the product about?"):
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_SECRET_KEY")

    index_name = "ai-ia"
    headers_to_split_on = [("##", "Header 2")]

    fp = open("data.md", encoding="utf-8")
    data = fp.read()
    fp.close()

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    md_header_splits = markdown_splitter.split_text(data)
    model_name = "text-embedding-3-large"  # Updated to a valid model name
    embeddings = OpenAIEmbeddings(model=model_name, api_key=OPENAI_API_KEY)

    vector_store = PineconeVectorStore.from_documents(
        md_header_splits, embeddings, index_name=index_name
    )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    similar_docs = vector_store.similarity_search(prompt)
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm, prompt=retrieval_qa_chat_prompt
    )
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
    )
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                        You are a helpful assistant, your name is {name} that returns information requested in a really short and
                        concise manner.
                """,
            ),
            (
                "user",
                """
                        What is the product about?
                """,
            ),
        ]
    )
    prompt = template.invoke({"name": "Bob"}).to_messages()

    result = retrieval_chain.invoke({"input": "Hello"})
    print("RAG Finished")
