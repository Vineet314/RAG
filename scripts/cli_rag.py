import chromadb
import os
from openai import OpenAI
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

# initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("API_KEY"))

# initialize ChromaDB client
db_dir = r"../chromadb"
chroma_client = chromadb.PersistentClient(path=db_dir)
collection = chroma_client.get_collection(name="my_collection")

def chat(query:str) -> str:
    # Embedify
    query_vector = openai_client.embeddings.create(input=query, model='text-embedding-3-small').data[0].embedding

    results = collection.query(query_embeddings=[query_vector], n_results=3)
    retrieved_chunks = results['documents'][0]
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    # prompt the LLM
    prompt = f"""
        You are a helpful assistant for answering questions based on a collection of research papers.
        Use the following retrieved context along with your own knowledge to answer the question.
        If you don't know the answer from the context, just say that you don't know.

        Context:
        {context}

        Question: {query}

        Answer:
        """
    print("RAG_AI: ", end="", flush=True)
    response_stream = openai_client.responses.create(model='gpt-4.1-mini', input=prompt, stream=True)
    for event in response_stream:
        if isinstance(event, ResponseTextDeltaEvent):
            print(event.delta, end='', flush=True)

if __name__ == '__main__':
    while True:
        # take a query from the user
        query = input("\n\nHuman: ")
        if query.lower().strip() in ('bye','exit','quit'):
            print("AI: Thank you for chatting with me")
            break
        chat(query)