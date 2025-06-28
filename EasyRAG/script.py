import gradio as gr
import os, io

from google import genai
from pathlib import Path

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# This will store uploaded PDF references for each session
session_pdfs = {}

def upload_pdfs(files):
    """Handles PDF uploads and stores the Gemini file references in memory."""
    global session_pdfs

    doc_data = [io.BytesIO(Path(file).read_bytes()) for file in files]
    pdf_refs = [client.files.upload(file=data, config={"mime_type": "application/pdf"}) for data in doc_data]
    session_id = id(files[0])  # Use unique session ID based on file object
    session_pdfs[session_id] = pdf_refs
    return "PDFs uploaded. You can now start chatting.", session_id

def chat_with_pdfs(message, history, session_id):
    """Handles chat using the uploaded PDFs."""
    pdfs = session_pdfs.get(session_id, [])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[*pdfs, message])
    
    return response.text

# Gradio Blocks app
with gr.Blocks() as demo:
    with gr.Row(scale=1):
        # column 1: the file picker
        with gr.Column(scale=2):
            file_input = gr.File(
                file_types=[".pdf"],
                file_count="multiple",
                label="Upload PDFs")

        # column 2: button on top, status textbox underneath
        with gr.Column(scale=1):
            upload_btn = gr.Button("Upload")
            upload_output = gr.Textbox(label="Status")

    session_state = gr.State()

    chat = gr.ChatInterface(
        fn=chat_with_pdfs,
        additional_inputs=[session_state],
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="Ask a question..."))

    upload_btn.click(
        upload_pdfs,
        inputs=file_input,
        outputs=[upload_output, session_state])

if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://localhost:3141", new=2)

    demo.launch(server_name='127.0.0.1', server_port=3141)
