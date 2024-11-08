import gradio as gr
import os
from helpers.config import config
from helpers.utils import reset_flags
from llm.llm import LLM
from ocr.ocr import load_models, OCR
from retriever.indexing import IndexDocument
from PIL import Image

# Load models
embed_model = IndexDocument.load_model()
LLM_MODEL, LLM_TOKENIZER = LLM.initialize_model()
ocr_models = load_models()
ocr = OCR(ocr_models)

# Upload and RAG functions
def upload_pdf(pdf_file, chunk_size, overlap):
    try:
        if pdf_file is None:
            return "Please upload a document.", gr.update(visible=False)
            
        ocr.extract_page(pdf_file, chunk_size, overlap)
        IndexDocument.index(embed_model)
        reset_flags()

        return "Uploaded successfully. Now you can ask.", gr.update(visible=True)
    except Exception as e:
        print(str(e))
        return "Failed to upload document.", gr.update(visible=False)

def rag_it(query: str, mood: str):
    try:
        if not query.strip():
            return [], "Please enter a question", []
            
        # Convert mood to lowercase to match your existing code
        mood = mood.lower()
        
        if config.faiss_index is None:
            return [], "Please upload a document first", []
            
        top_3 = IndexDocument.search(query, embed_model, config.faiss_index, config.docs_mapping)

        response, refrences, doc_id = LLM.generate_response(
            model=LLM_MODEL, 
            tokenizer=LLM_TOKENIZER, 
            mood=mood, 
            query=query, 
            documents=top_3
        )        
        
        if not refrences or not doc_id:
            return [(query, response)], "Answered (no citations available)", []
            
        images_paths = [
            os.path.join("output", doc_id, "chunks_layout", f"chunk_layout_{ref}.png")
            for ref in refrences
        ]
        
        citation_images = []
        for path in images_paths:
            try:
                citation_images.append(Image.open(path))
            except Exception as e:
                print(f"Failed to load image: {path}, Error: {str(e)}")

        return [(query, response)], "Answered Successfully", citation_images
    except Exception as e:
        print(f"Error in rag_it: {str(e)}")
        return [], "Couldn't answer", []

# Custom CSS with lighter header
css = """
body, html {
    margin: 0;
    padding: 0;
    background-color: #f0f6ff !important;
    font-family: 'Inter', sans-serif;
}

.container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

.gradio-container {
    margin: 0 !important;
    padding: 0 !important;
}

/* Header styling - lighter version */
.header {
    background: linear-gradient(135deg, #63B3ED 0%, #4299E1 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 10px;
    color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Group styling */
.gr-group {
    background-color: white !important;
    padding: 20px !important;
    border-radius: 15px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    margin: 10px !important;
}

/* Button styling */
.gr-button {
    background: linear-gradient(135deg, #63B3ED 0%, #4299E1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: transform 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
}

/* File upload styling */
.gr-file {
    border: 2px dashed #4299e1 !important;
    border-radius: 10px !important;
    padding: 20px !important;
}

/* Slider styling */
.gr-slider {
    margin: 10px 0 !important;
}

/* Chat container styling */
.chatbot {
    height: 500px !important;
    background-color: white !important;
    border-radius: 15px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

/* Gallery styling */
.gr-gallery {
    border-radius: 10px !important;
    overflow: hidden !important;
    background-color: white !important;
}

/* Status box styling */
.gr-textbox {
    border-radius: 8px !important;
    margin-top: 10px !important;
}
"""

def gradio_interface():
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📚 Jisr Document Assistant
            ### Intelligent Document Analysis and Question Answering System
            """,
            elem_classes="header"
        )
        
        with gr.Row():
            # Left Column
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📄 Document Upload")
                    files = gr.File(
                        label="Upload your PDF or Text document",
                        file_types=[".pdf", ".txt"],
                        file_count="single",
                        type="filepath"
                    )
                    
                    with gr.Row():
                        chunk_size = gr.Slider(
                            100, 2000, value=650, step=50,
                            label="Chunk Size"
                        )
                        overlap = gr.Slider(
                            0, 400, value=200, step=50,
                            label="Overlap Size"
                        )
                    
                    status = gr.Textbox(label="Status")
                
                with gr.Group():
                    gr.Markdown("### 🔍 Citations")
                    sources_output = gr.Gallery(
                        label="Document References",
                        columns=2,
                        height=400
                    )
            
            # Right Column
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 💬 Chat Interface")
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        avatar_images=(
                            "https://i.ibb.co/tKqTKC8/office-man.png",
                            "https://i.ibb.co/qmjN6Rj/images-2.png"
                        )
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Ask a question",
                            placeholder="Type your question here...",
                            scale=8
                        )
                        rag_button = gr.Button(
                            "Ask 🔎",
                            visible=False,  # Initially hidden
                            scale=1
                        )
                    
                    choice = gr.Radio(
                        ["rag", "summary"],
                        label="Response Mode",
                        value="rag"
                    )

        # Event handlers
        files.upload(
            fn=upload_pdf,
            inputs=[files, chunk_size, overlap],
            outputs=[status, rag_button]
        )
        
        rag_button.click(
            fn=rag_it,
            inputs=[msg, choice],
            outputs=[chatbot, status, sources_output]
        )
        
        msg.submit(
            fn=rag_it,
            inputs=[msg, choice],
            outputs=[chatbot, status, sources_output]
        )

    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch(share=True)