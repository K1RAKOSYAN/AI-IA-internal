from os import getenv
from initialize import initialize
def TestEnvironmentVariables():
    PINECONE_API_KEY = getenv("PINECONE_API_KEY")
    OPENAI_SECRET_KEY = getenv("OPENAI_SECRET_KEY")
    
    assert OPENAI_SECRET_KEY != None
    assert PINECONE_API_KEY != None

if __name__ == "__main__":
    initialize()
    TestEnvironmentVariables()
    print("Everything passed")