import os
import argparse
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from openai import OpenAI
from openai.types.responses.response_created_event import ResponseCreatedEvent

# ------------------defaults------------------
oclient = OpenAI(api_key=os.getenv("API_KEY"))
db_dir = r"../chromadb"
name = "my_collection"
# this script assumes a ChromaDB collection has already been created 
# If not, you can run the script knowledge-base.py to make one

prev_id = None # initialize it to none
# ------------------utils------------------
def chat(query:str, collection:Collection, num_results:int=3) -> None:
    global prev_id
    
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
    response_stream = oclient.responses.create(model='gpt-4.1-mini', input=prompt, stream=True, previous_response_id=prev_id)
    for event in response_stream:
        if hasattr(event, "delta"):
            print(event.delta, end='', flush=True)
        if isinstance(event, ResponseCreatedEvent):
            prev_id = event.response.id

def main(args):
    cclient = PersistentClient(path=args.db_dir)
    collection = cclient.get_collection(name=args.name)
    while True:
        # take a query from the user
        query = input("\n\nHuman: ")
        if query.lower().strip() in ('bye','exit','quit'):
            print("RAG_AI: Thank you for chatting with me")
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