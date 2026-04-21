from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
response = client.embeddings.create(
    input = "Hey! This is a text ready for embedding",
    model = "text-embedding-3-small"
)

print(response)