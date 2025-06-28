import os
import argparse
import chromadb
import numpy as np
from openai import OpenAI
# from google import genai

# ------------------defaults------------------
oclient = OpenAI(api_key=os.getenv("API_KEY"))
# gclient = genai.client(api_key=os.getenv("GEMINI_API_KEY"))

# Key-word arguments, can be over-ridden via CLI
text_file = r"../data/knowledge.txt"
chunk_size=3500
chunk_overlap=300
save_embds = False
load_embds = False
load_path = None
save_path = None
db_dir = r"chromadb/"
name = "my_collection"

#------------------utils------------------
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Splits a long text into smaller, overlapping chunks.

    Args:
        text: The input text to be chunked.
        chunk_size: The desired size of each chunk.
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def chunks2embds(chunks:list[str]) -> np.ndarray:

    if (s := sum(len(i) for i in chunks) / 3.5) > 300000: # obeying rate limits
        n = int(np.ceil(s/300000))
        k, m = divmod(len(chunks), n)
        chunks_ = [chunks[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    embds = []
    for hehe in chunks_:
        resp = oclient.embeddings.create(
            input=hehe,
            model="text-embedding-3-small")
        # will add support for gemini embedding models in a future commit 
        embds.extend([resp.data[i].embedding for i in range(len(hehe))])

    return np.array(embds)

def make_collection(db_dir:str, chunks:list[str], embds:np.ndarray, name:str) -> None:
    os.makedirs(db_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_dir)
    collection = chroma_client.get_or_create_collection(name=name)
    collection.add(ids=[f'chunk{i}' for i in range(len(chunks))], documents=chunks, embeddings=embds)

def main(**kwargs):
    # read from the text file 
    text = open(kwargs['text_file'], mode='r',encoding='utf-8').read()
    # chunk the text
    chunks = chunk_text(text, kwargs["chunk_size"], kwargs["chunk_overlap"])
    # get embeddings
    if kwargs["load_embds"]:
        assert kwargs["load_path"] is not None, "Provide a path to load embeddings from"
        embds = np.load(kwargs["load_path"], 'r')
    else: 
        embds = chunks2embds(chunks)
    # save if you want to
    if kwargs["save_embds"]:        
        assert kwargs["save_path"] is not None, "Provide a path to save embeddings to"
        np.save(kwargs["save_path"], embds)
    # make vectore store
    make_collection(db_dir=kwargs["db_dir"], chunks=chunks, embds=embds, name=kwargs["name"])

def parse_args():
    parser = argparse.ArgumentParser(description="Make a knowledge base using ChromaDB")
    parser.add_argument('--text_file', type=str, default=text_file, help='Path to the text file containing knowledge')
    parser.add_argument('--chunk_size', type=int, default=chunk_size, help='Size of each text chunk')
    parser.add_argument('--chunk_overlap', type=int, default=chunk_overlap, help='Overlap size between text chunks')
    parser.add_argument('--save_embds', action='store_true', help='Save embeddings to a file')
    parser.add_argument('--load_embds', action='store_true', help='Load embeddings from a file')
    parser.add_argument('--load_path', type=str, default=load_path, help='Path to load embeddings from')
    parser.add_argument('--save_path', type=str, default=save_path, help='Path to save embeddings to')
    parser.add_argument('--db_dir', type=str, default=db_dir, help='Directory for ChromaDB')
    parser.add_argument('--name', type=str, default=name, help='Name of the ChromaDB collection')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    main(**vars(args))