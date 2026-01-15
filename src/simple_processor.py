"""
Optimized Single-Pass Document Processor
Reduces API calls and memory usage
"""
from typing import List, Dict, Any
from PIL import Image
import json
import re
import logging

logger = logging.getLogger(__name__)


class SimpleDocumentProcessor:
    """Streamlined processor - fewer API calls, faster processing"""
    
    def __init__(self, qwen_client):
        self.qwen_client = qwen_client
        logger.info("✅ Initialized Simple Document Processor")
    
    def process_document(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Process document in 2 calls instead of 5+"""
        
        logger.info(f"🚀 Processing {len(images)} page(s) - Optimized pipeline")
        
        # STEP 1: Single-pass extraction (instead of classify + extract separately)
        extractions = self._extract_all_pages(images)
        
        # STEP 2: Combine and validate (if multi-page)
        if len(images) == 1:
            final_result = self._format_single_page(extractions[0])
        else:
            final_result = self._combine_pages(extractions)
        
        logger.info(f"✅ Processing complete - Confidence: {final_result.get('confidence', 0):.2f}")
        
        return final_result
    
    def _extract_all_pages(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Extract from all pages efficiently"""
        
        results = []
        
        for i, image in enumerate(images, 1):
            logger.info(f"📄 Processing page {i}/{len(images)}...")
            
            # Single comprehensive prompt
            prompt = f"""Analyze this document page {i} and extract ALL information in JSON format.

Provide:
{{
  "document_type": "[invoice/receipt/form/report/research_paper/letter/other]",
  "confidence": 0.0-1.0,
  "text_content": "all readable text",
  "key_fields": {{
    "field_name": "value"
  }},
  "tables": [
    {{"description": "table description", "data": []}}
  ],
  "amounts": {{"subtotal": "", "tax": "", "total": ""}},
  "dates": [],
  "important_info": []
}}

Extract EVERYTHING you see. Return ONLY valid JSON, no explanation."""

            result = self.qwen_client.query(
                text=prompt,
                images=[image],
                max_tokens=800,  # Reduced from 3000
                temperature=0.1
            )
            
            if result['success']:
                parsed = self._parse_response(result['content'])
                parsed['page_number'] = i
                results.append(parsed)
            else:
                logger.error(f"Page {i} failed: {result['error']}")
                results.append({
                    'page_number': i,
                    'error': result['error']
                })
        
        return results
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response"""
        try:
            # Remove markdown code blocks
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            
            # Find JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {'raw_content': response}
        except Exception as e:
            logger.warning(f"JSON parse failed: {e}")
            return {'raw_content': response}
    
    def _format_single_page(self, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Format single page result"""
        
        return {
            'document_type': extraction.get('document_type', 'unknown'),
            'total_pages': 1,
            'confidence': extraction.get('confidence', 0.7),
            'extracted_content': extraction,
            'processing_time': 'fast',
            'status': 'success'
        }
    
    def _combine_pages(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multi-page results simply"""
        
        # Get document type from first page
        doc_type = extractions[0].get('document_type', 'unknown')
        
        # Calculate average confidence
        confidences = [e.get('confidence', 0.5) for e in extractions if 'confidence' in e]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.7
        
        # Combine all content
        combined = {
            'document_type': doc_type,
            'pages': extractions
        }
        
        return {
            'document_type': doc_type,
            'total_pages': len(extractions),
            'confidence': avg_confidence,
            'extracted_content': combined,
            'processing_time': 'optimized',
            'status': 'success'
        }


def test_simple_processor():
    """Test optimized processor"""
    print("=" * 70)
    print("Testing Optimized Processor")
    print("=" * 70)
    
    from models.qwen_client import Qwen3VLClient
    from PIL import ImageDraw
    
    # Initialize
    print("\n[1/2] Initializing...")
    client = Qwen3VLClient(timeout=60)  # Shorter timeout
    processor = SimpleDocumentProcessor(client)
    print("      ✅ Ready")
    
    # Create test doc
    print("\n[2/2] Creating and processing test document...")
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "INVOICE #12345", fill='black')
    draw.text((50, 100), "Date: 2025-01-15", fill='black')
    draw.text((50, 150), "Total: $999.99", fill='black')
    
    # Process
    import time
    start = time.time()
    result = processor.process_document([img])
    elapsed = time.time() - start
    
    print(f"\n✅ Processed in {elapsed:.1f} seconds")
    print(f"📊 Type: {result['document_type']}")
    print(f"🎯 Confidence: {result['confidence']:.2f}")
    print(f"\nExtracted:\n{json.dumps(result['extracted_content'], indent=2)[:300]}...")
    
    return True


if __name__ == "__main__":
    test_simple_processor()