import os
import argparse
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection as chromadb_collection
from openai import OpenAI
# from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

# ------------------defaults------------------
oclient = OpenAI(api_key=os.getenv("API_KEY"))
db_dir = r"../chromadb"
name = "my_collection"
# this script assumes a ChromaDB collection has already been created 
# If not, you can run the script knowledge-base.py to make one

# ------------------utils------------------
def chat(query:str, collection:chromadb_collection, num_results:int=3) -> None:
    # Embedify
    query_vector = oclient.embeddings.create(input=query, model='text-embedding-3-small').data[0].embedding

    results = collection.query(query_embeddings=[query_vector], n_results=num_results)
    retrieved_chunks = results['documents'][0]
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    # prompt the LLM
    prompt = f"""
        You are a helpful assistant for answering questions based on a collection of documents.
        Use the following retrieved context along with your own knowledge to answer the question.
        If you don't know the answer from the context, just say that you don't know.

        Context:
        {context}

        Question: \n{query}

        Answer:
        """
    print("RAG_AI: ", end="", flush=True)
    response_stream = oclient.responses.create(model='gpt-4.1-mini', input=prompt, stream=True)
    for event in response_stream:
        if hasattr(event, "delta"):
            print(event.delta, end='', flush=True)
        # if isinstance(event, ResponseTextDeltaEvent):
        #     print(event.delta, end='', flush=True)

def main(args):
    cclient = PersistentClient(path=args.db_dir)
    if args.name not in cclient.list_collections():
        raise FileNotFoundError(f"Collection '{name}' not found. Please create it first.")
    collection = cclient.get_collection(name=args.name)
    while True:
        # take a query from the user
        query = input("\n\nHuman: ")
        if query.lower().strip() in ('bye','exit','quit'):
            print("AI: Thank you for chatting with me")
            break
        chat(query, collection)

def parse_args():
    parser = argparse.ArgumentParser(description="Run a RAG chat session with a ChromaDB collection.")
    parser.add_argument('--db_dir', type=str, default=db_dir, help='Directory for the ChromaDB database.')
    parser.add_argument('--name', type=str, default=name, help='Name of the ChromaDB collection to use.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)