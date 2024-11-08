import os
import threading
import queue
import gradio as gr

from helpers.config import config
from helpers.utils import reset_flags
from llm.llm import LLM
from ocr.ocr import load_models, OCR
from retriever.indexing import IndexDocument
from dotenv import load_dotenv
from PIL import Image


# load embedding model
embed_model = IndexDocument.load_model()

# load LLM
LLM_MODEL, LLM_TOKENIZER = LLM.initialize_model()

# load OCR
ocr_models = load_models()
ocr = OCR(ocr_models)


def upload(files: list, chunk_size: int, chunk_overlap: int):
    try:
        ocr.extract_page(files, chunk_size, chunk_overlap)
        
        IndexDocument.index(embed_model)

        reset_flags()

        return "Uploaded successfully. Now you can ask.", gr.update(visible=True)
    except Exception as e:
        print(str(e))
        return "Failed to upload document."



def rag_it(query: str, mood: str):
    try:
        top_3 = IndexDocument.search(query, embed_model, config.faiss_index, config.docs_mapping)

        response, refrences, doc_id = LLM.generate_response(
            model=LLM_MODEL, tokenizer=LLM_TOKENIZER, mood=mood, query=query, documents=top_3
        )        
        #load chunks_layout images using doc_id, chunk_id in refrences
        images_paths = [
            os.path.join("output", doc_id, "chunks_layout", f"chunk_layout_{ref}.png")
            for ref in refrences
            ]
        #load images
        citation_images = [Image.open(path) for path in images_paths]
        

        return [(query, response)], "Answered Successfully", citation_images
    except Exception as e:
        print(str(e))
        return [], "Couldn't answer", []
    
def upload_pdf(pdf_file, chunk_size, overlap):
    try:
        ocr.extract_page(pdf_file, chunk_size, overlap)
        
        IndexDocument.index(embed_model)

        reset_flags()

        return "Uploaded successfully. Now you can ask.", gr.update(visible=True)
    except Exception as e:
        print(str(e))
        return "Failed to upload document."

css = """
body, html {
    margin: 0;
    padding: 0;
    background-color: #F8F9FA; /* Light gray background */
    font-family: 'Roboto', sans-serif; /* Matching font */
    overflow-x: hidden;
}

.container, .main, .wrap {
    max-width: 100% !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gr-button {
    margin: 5px !important;
    font-size: 0.8em !important;
    padding: 5px 10px !important;
    border-radius: 5px !important;
}
"""

css = """
body, html {
    margin: 0;
    padding: 0;
    background-color: #F8F9FA; /* Light gray background */
    font-family: 'Roboto', sans-serif; /* Matching font */
    overflow-x: hidden;
}

.container, .main, .wrap {
    max-width: 100% !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gr-button {
    min-width: 60px !important;
    height: 36px !important;
    line-height: 1 !important;
    font-size: 14px !important;
    padding: 0 10px !important;
    margin-top: 20px !important; /* Adjust this value to align with your text input */
}
"""

def gradio_interface(theme=gr.themes.Base(), css=css):
    with gr.Blocks(fill_height=True, fill_width=True) as demo:
        gr.Markdown("# Jisr RAG ")
        
        with gr.Row():
            with gr.Column(scale=1):
                uploading = gr.Group()
                with uploading:
                    files = gr.File( label="Upload Document",
                    file_types=[".pdf", ".txt"],  # Add dots before extensions
                    file_count="single",          # Specify single file upload
                    type="filepath" )
                    print("files:------------------", files)
                    chunk_size = gr.Slider(100, 2000, value=650, step=50, label="Chunk size")
                    overlap = gr.Slider(0, 400, value=200, step=50, label="Overlap")
                    upload_button = gr.Button("Upload", visible=False)
                gr.Image("https://i.ibb.co/kydxNm1/images-3.png", height=200, show_download_button=False, show_label=False, show_fullscreen_button=False, show_share_button=False)
                    
                
            sources_output = gr.Gallery(label="citation", scale=3)
            with gr.Column(scale=3):
                chat_interface = gr.Group()
                with chat_interface:
                    chatbot = gr.Chatbot(label="Conversation", height=300, layout="panel", avatar_images=("https://i.ibb.co/tKqTKC8/office-man.png", "https://i.ibb.co/qmjN6Rj/images-2.png"))
                    msg = gr.Textbox(label="Ask a question",)
                rag_button = gr.Button("RAG it now!", visible=False)
                choice = gr.Radio(["rag", "summary"], label="Select a behavior", value="rag")
                status = gr.Textbox(label="Status")
                
            
        files.upload(upload_pdf, inputs=[files, chunk_size, overlap], outputs=[status, rag_button])
        upload_button.click(
            upload,
            inputs=[files, chunk_size, overlap],
            outputs=[status, rag_button])
    
       
       
        rag_button.click(rag_it, 
                         inputs=[msg, choice], 
                         outputs=[chatbot, status, sources_output])
        
        msg.submit(rag_it, inputs=[msg, choice], outputs=[chatbot, status, sources_output])
 

    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch(share=True)