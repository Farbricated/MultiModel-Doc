"""
Document Processing Utilities
Handles PDF, images, and document conversions
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image
import PyPDF2
import pdfplumber
from pdf2image import convert_from_path
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process various document types into images and text"""
    
    def __init__(
        self,
        dpi: int = 200,
        poppler_path: Optional[str] = None
    ):
        """
        Args:
            dpi: Resolution for PDF to image conversion
            poppler_path: Path to poppler binaries (if not in PATH)
        """
        self.dpi = dpi
        self.poppler_path = poppler_path
        
        logger.info(f"Initialized DocumentProcessor (DPI: {dpi})")
    
    def is_supported(self, file_path: Union[str, Path]) -> bool:
        """Check if file type is supported"""
        file_path = Path(file_path)
        supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        return file_path.suffix.lower() in supported_extensions
    
    def pdf_to_images(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        """Convert PDF pages to images"""
        try:
            pdf_path = Path(pdf_path)
            logger.info(f"Converting PDF to images: {pdf_path.name}")
            
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                poppler_path=self.poppler_path
            )
            
            logger.info(f"Converted {len(images)} pages")
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert PDF: {e}")
            raise
    
    def extract_text_pdfplumber(self, pdf_path: Union[str, Path]) -> List[str]:
        """Extract text from PDF using pdfplumber"""
        try:
            pdf_path = Path(pdf_path)
            texts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    texts.append(text)
            
            logger.info(f"Extracted text from {len(texts)} pages")
            return texts
            
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return []
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load an image file"""
        try:
            image_path = Path(image_path)
            img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            logger.debug(f"Loaded image: {image_path.name} ({img.size})")
            return img
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
    
    def process_document(
        self,
        file_path: Union[str, Path]
    ) -> Tuple[List[Image.Image], List[str]]:
        """Process document into images and text"""
        file_path = Path(file_path)
        
        if not self.is_supported(file_path):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        logger.info(f"Processing document: {file_path.name}")
        
        if file_path.suffix.lower() == '.pdf':
            images = self.pdf_to_images(file_path)
            texts = self.extract_text_pdfplumber(file_path)
            return images, texts
        else:
            img = self.load_image(file_path)
            return [img], []


def test_document_processor():
    """Test document processor"""
    print("=" * 70)
    print("Testing Document Processor")
    print("=" * 70)
    
    processor = DocumentProcessor()
    
    print("\n[1/2] Creating test image...")
    test_img = Image.new('RGB', (800, 600), color='white')
    test_path = Path("data/input/test_image.png")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_img.save(test_path)
    print(f"      ✅ Created: {test_path}")
    
    print("\n[2/2] Testing image loading...")
    try:
        loaded_img = processor.load_image(test_path)
        print(f"      ✅ Loaded image: {loaded_img.size}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ Document Processor Ready!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    import sys
    success = test_document_processor()
    sys.exit(0 if success else 1)