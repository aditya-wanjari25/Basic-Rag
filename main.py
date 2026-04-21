from openai import OpenAI
from dotenv import load_dotenv
from sample import TEXT
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

def build_index(chunks):
    index = []
    for chunk in chunks:
        d = dict()
        d[chunk] = embed_text(chunk)
        index.append(d)
    return index


def cosine_sim(vecA, vecB):
    dot_prod = 0
    for i in range(len(vecA)):
        dot_prod += vecA[i] * vecB[i]
    
    vecA_mag = 0
    for i in range(len(vecA)):
        vecA_mag += vecA[i]**2
    
    vecA_mag  = vecA_mag**0.5
    
    vecB_mag = 0
    for i in range(len(vecB)):
        vecB_mag += vecB[i]**2
    
    vecB_mag = vecB_mag ** 0.5

    cosine_sim = dot_prod / (vecA_mag * vecB_mag)
    return cosine_sim

def retrieve(query, index, top_k):
    query_embed = embed_text(query)
    scores = []
    for i in range(len(index)):
        d = dict()
        item = index[i]
        key, value = list(item.items())[0]
        d[key] = cosine_sim(query_embed, value)
        scores.append(d)

    result = sorted(
    scores,
    key=lambda x: list(x.values())[0],
    reverse=True
    )[:top_k]

    return result


chunks = chunk_text(TEXT,50,15)
index = build_index(chunks)
query = "explain scoring system in tennis"
scores = retrieve(query,index,5)
print(scores)
