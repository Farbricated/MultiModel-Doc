"""
Ultra-fast processor for competition demo
Prioritizes speed over detail
"""
from typing import List, Dict, Any
from PIL import Image
import json
import re
import logging

logger = logging.getLogger(__name__)


class FastDocumentProcessor:
    """Lightning-fast processor for demos"""
    
    def __init__(self, qwen_client):
        self.qwen_client = qwen_client
        logger.info("⚡ Initialized Fast Document Processor")
    
    def process_document(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Process document FAST"""
        
        logger.info(f"⚡ Fast processing {len(images)} page(s)")
        
        # Process only first page for speed (or limit to 3 pages max)
        pages_to_process = images[:min(3, len(images))]
        
        extractions = []
        for i, image in enumerate(pages_to_process, 1):
            logger.info(f"📄 Page {i}...")
            extraction = self._extract_page_fast(image, i)
            extractions.append(extraction)
        
        # Simple combination
        result = self._combine_simple(extractions, len(images))
        
        logger.info(f"✅ Done - Confidence: {result['confidence']:.2f}")
        return result
    
    def _extract_page_fast(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """Fast extraction with short prompt"""
        
        # Ultra-concise prompt
        prompt = f"""Analyze this document. Return JSON only:

{{
  "type": "invoice/receipt/form/table/report/letter/other",
  "confidence": 0.9,
  "main_content": "brief summary",
  "key_data": {{"field": "value"}},
  "amounts": {{"total": ""}},
  "dates": [""]
}}

Be concise. JSON only, no explanation."""

        result = self.qwen_client.query(
            text=prompt,
            images=[image],
            max_tokens=500,  # Very low for speed
            temperature=0.1
        )
        
        if result['success']:
            parsed = self._parse_json(result['content'])
            parsed['page'] = page_num
            parsed['success'] = True
            return parsed
        else:
            return {
                'page': page_num,
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'type': 'unknown',
                'confidence': 0.0
            }
    
    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Quick JSON parse"""
        try:
            # Clean response
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = re.sub(r'```json\s*|\s*```', '', response)
            
            # Extract JSON
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            
            # Fallback
            return {
                'type': 'document',
                'confidence': 0.6,
                'main_content': response[:200]
            }
        except:
            return {
                'type': 'document',
                'confidence': 0.5,
                'raw': response[:200]
            }
    
    def _combine_simple(self, extractions: List[Dict], total_pages: int) -> Dict[str, Any]:
        """Simple fast combination"""
        
        # Get type from first successful extraction
        doc_type = 'document'
        for ext in extractions:
            if ext.get('success') and 'type' in ext:
                doc_type = ext['type']
                break
        
        # Average confidence
        confidences = [e.get('confidence', 0.5) for e in extractions if e.get('success')]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.6
        
        return {
            'document_type': doc_type,
            'total_pages': total_pages,
            'processed_pages': len(extractions),
            'confidence': round(avg_conf, 2),
            'extracted_content': {
                'pages': extractions
            },
            'status': 'success',
            'note': 'Fast mode - first 3 pages only' if total_pages > 3 else 'Fast mode'
        }


def test_fast():
    """Test fast processor"""
    print("=" * 70)
    print("Testing FAST Processor")
    print("=" * 70)
    
    from models.qwen_client import Qwen3VLClient
    from PIL import ImageDraw
    import time
    
    client = Qwen3VLClient(timeout=120)
    processor = FastDocumentProcessor(client)
    
    # Simple test
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "INVOICE #123", fill='black')
    draw.text((20, 50), "Total: $500", fill='black')
    
    start = time.time()
    result = processor.process_document([img])
    elapsed = time.time() - start
    
    print(f"\n✅ Processed in {elapsed:.1f}s")
    print(f"Type: {result['document_type']}")
    print(f"Confidence: {result['confidence']}")
    print(f"\n{json.dumps(result, indent=2)[:400]}...")


if __name__ == "__main__":
    test_fast()