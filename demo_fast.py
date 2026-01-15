"""
Ultra-fast demo for competition
"""
import sys
from pathlib import Path
import gradio as gr
import json
from PIL import Image
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.qwen_client import Qwen3VLClient
from utils.document_processor import DocumentProcessor as DocUtils
from fast_processor import FastDocumentProcessor


print("⚡ Initializing FAST Document Intelligence...")
client = Qwen3VLClient(timeout=120)
doc_utils = DocUtils(dpi=100)
processor = FastDocumentProcessor(client)
print("✅ Ready!\n")


def process_file(file, progress=gr.Progress()):
    """Process with progress bar"""
    
    if file is None:
        return "⚠️ Upload a document", "{}", 0.0, 0.0
    
    try:
        progress(0, desc="Loading document...")
        start = time.time()
        
        file_path = Path(file.name)
        
        # Load
        if file_path.suffix.lower() == '.pdf':
            images, _ = doc_utils.process_document(file_path)
        else:
            images = [doc_utils.load_image(file_path)]
        
        progress(0.3, desc="Processing with AI...")
        
        # Process
        result = processor.process_document(images)
        
        progress(1.0, desc="Done!")
        elapsed = time.time() - start
        
        # Format
        doc_type = result.get('document_type', 'unknown')
        confidence = result.get('confidence', 0.0)
        pages = result.get('total_pages', 0)
        
        status = f"✅ **{pages} page(s)** processed in **{elapsed:.1f}s** | Type: **{doc_type.upper()}**"
        
        extracted = json.dumps(result.get('extracted_content', {}), indent=2)
        
        return status, extracted, confidence, elapsed
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "{}", 0.0, 0.0


# Fixed for Gradio 6.0
demo = gr.Blocks()

with demo:
    
    gr.Markdown("""
    # ⚡ Multi-Modal Document Intelligence
    **Fast Mode** | Qwen3-VL-4B | Optimized for Competition Demo
    """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="📄 Upload", file_types=[".pdf", ".png", ".jpg"])
            process_btn = gr.Button("⚡ Process Fast", variant="primary")
            
            confidence_out = gr.Slider(label="🎯 Confidence", minimum=0, maximum=1, interactive=False)
            time_out = gr.Number(label="⏱️ Time (s)", precision=1)
        
        with gr.Column():
            status_out = gr.Markdown("Ready...")
            json_out = gr.Code(label="📝 Extracted (JSON)", language="json", lines=15)
    
    process_btn.click(
        fn=process_file,
        inputs=[file_input],
        outputs=[status_out, json_out, confidence_out, time_out]
    )

if __name__ == "__main__":
    # Fixed launch for Gradio 6.0
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )