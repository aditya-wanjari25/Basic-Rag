from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def embed_text(text):

    client = OpenAI()
    response = client.embeddings.create(
        input = text,
        model = "text-embedding-3-small"
    )
    return response.data[0].embedding

def chunk_text(text, chunk_size, overlap):
    chunks = []
    flag = 1
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i: i+chunk_size])

    return chunks

text = "This is a text about to be chunked"
print(chunk_text(text,3,1))