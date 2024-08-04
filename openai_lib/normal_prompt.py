from openai import OpenAI
from os import getenv

async def run_query():
    OPENAI_API_KEY = getenv("OPENAI_SECRET_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
        ]
    )

    print("Completions Finished.")
