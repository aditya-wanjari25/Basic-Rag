from openai import OpenAI
from dotenv import load_dotenv
from sample import TEXT
import math
from collections import defaultdict

load_dotenv()
client = OpenAI()

def embed_text(text):

    response = client.embeddings.create(
        input = text,
        model = "text-embedding-3-small"
    )
    return response.data[0].embedding

def chunk_text(text, chunk_size, overlap):
    chunks = []
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

def invoke_llm(query, context):
    PROMPT = f""" you are a helpful assistant. Answer questions based only based on context provided. 
    If you cannot find answer in context, say 'I dont know man'. 

    question = {query};
    context = {context}
     """
    
    response = client.responses.create(
        model = "gpt-4o-mini",
        input = PROMPT
    )
    return response.output_text

def generate(query):
    chunks = chunk_text(TEXT,50,15)
    index = build_index(chunks)
    scores = retrieve(query,index,5)
    print(invoke_llm(query, scores))
    print(scores)

def bm25_score(query, chunks):
    query_tokens = query.lower().split()
    chunk_tokens = [chunk.lower().split() for chunk in chunks]
    total_chunks = len(chunk_tokens)
    avgdl = sum(len(chunk) for chunk in chunk_tokens) / total_chunks
    scores = [0.0] * total_chunks
    k1 = 1.5
    b = 0.75

    for query_token in query_tokens:
        document_frequency = sum(1 for chunk in chunk_tokens if query_token in chunk)    
        inverse_doc_frequency = math.log(((total_chunks - document_frequency + 0.5)/(document_frequency + 0.5))+ 1)
        
        # score each chunk for this token
        for i, chunk in enumerate(chunk_tokens):
            tf = chunk.count(query_token)
            if tf == 0:
                continue
            
            doc_len = len(chunk)
            tf_normalized = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
            scores[i] += inverse_doc_frequency * tf_normalized

    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

chunks = [
    "BM25 is a ranking function used in information retrieval",
    "The quick brown fox jumps over the lazy dog",
    "BM25 uses term frequency and inverse document frequency",
    "Information retrieval systems use ranking to find relevant documents",
]

query = "BM25 ranking information retrieval"
results = bm25_score(query, chunks)

for idx, score in results:
    print(f"[{score:.2f}] chunk {idx}: {chunks[idx]}")

# query = "name surfaces that tennis is played on?"
# generate(query)