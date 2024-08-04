import os
import time
import asyncio
from initialize import initialize
from pinecone import Pinecone, ServerlessSpec
from vc_embeddings.chunker import chunkEmbeddingAndUpsert, queryEmbedding
from openai_lib.normal_prompt import run_query
from openai import OpenAI

def main():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_SECRET_KEY")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # This part is not necessary for asynchronous concurrent execution
    # and should be removed or handled differently if needed.

async def speed_test():
    start_time = time.time()

    # Run both functions concurrently
    await asyncio.gather(
        measure_time(queryEmbedding, "Langchain"),
        measure_time(run_query, "OpenAI completions")
    )

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.6f} seconds")

async def measure_time(coroutine, name):
    print(f"Running {name}")
    start = time.time()
    await coroutine()
    end = time.time()
    duration = end - start
    print(f"{name} took: {duration:.6f} seconds")
    return duration

if __name__ == "__main__":
    initialize()

    st = time.time()
    asyncio.run(speed_test())
    et = time.time()
    print(f"Total time taken: {et-st}")

# No need to call main here since we're focusing on the asynchronous part
