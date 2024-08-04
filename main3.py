import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# This is the YouTube video we're going to use.
YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=cdiD-9MMpb0"


model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")