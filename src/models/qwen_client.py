"""
Qwen3-VL-4B Client for LM Studio
Handles multi-modal (text + image) requests
"""
import base64
import requests
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
import io
import logging
import os

# Disable PaddleOCR connectivity check
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qwen3VLClient:
    """Client for Qwen3-VL-4B model via LM Studio"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model_name: str = "qwen3vl-4b",
        timeout: int = 240
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()
        
        logger.info(f"Initialized Qwen3VL Client: {base_url}")
    
    def _image_to_base64(self, image_input: Union[str, Path, Image.Image]) -> str:
        """Convert image to base64 string"""
        try:
            if isinstance(image_input, (str, Path)):
                # Load from file
                with open(image_input, 'rb') as f:
                    img_bytes = f.read()
                return base64.b64encode(img_bytes).decode('utf-8')
            
            elif isinstance(image_input, Image.Image):
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                image_input.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {e}")
            raise
    
    def query(
        self,
        text: str,
        images: Optional[List[Union[str, Path, Image.Image]]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query Qwen3-VL-4B with text and optional images
        
        Args:
            text: Text prompt/question
            images: List of images (file paths or PIL Images)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Enable streaming response
            
        Returns:
            Dict with response and metadata
        """
        try:
            # Build message content
            content = [{"type": "text", "text": text}]
            
            # Add images if provided
            if images:
                for img in images:
                    img_b64 = self._image_to_base64(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    })
            
            # Make API request
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
                **kwargs
            }
            
            logger.debug(f"Sending request with {len(images) if images else 0} images")
            
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                
                return {
                    "success": True,
                    "content": content,
                    "model": result.get('model', self.model_name),
                    "usage": result.get('usage', {}),
                    "raw_response": result
                }
            else:
                logger.error(f"Unexpected response format: {result}")
                return {
                    "success": False,
                    "content": "",
                    "error": "Unexpected response format"
                }
        
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after {self.timeout}s")
            return {
                "success": False,
                "content": "",
                "error": "Request timeout"
            }
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {
                "success": False,
                "content": "",
                "error": str(e)
            }
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                "success": False,
                "content": "",
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """Test if LM Studio is accessible"""
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return [m['id'] for m in data.get('data', [])]
            return []
        except:
            return []


# Test function
def test_qwen_client():
    """Test the Qwen3-VL client"""
    print("=" * 70)
    print("Testing Qwen3-VL-4B Client")
    print("=" * 70)
    
    client = Qwen3VLClient()
    
    # Test 1: Connection
    print("\n[1/4] Testing connection...")
    if client.test_connection():
        print("      ✅ Connected to LM Studio")
    else:
        print("      ❌ Cannot connect to LM Studio")
        print("      → Make sure LM Studio is running on localhost:1234")
        return False
    
    # Test 2: Available models
    print("\n[2/4] Checking available models...")
    models = client.get_available_models()
    if models:
        print(f"      ✅ Available: {models}")
    else:
        print("      ⚠️  No models found")
    
    # Test 3: Text-only query
    print("\n[3/4] Testing text-only query...")
    result = client.query("What is 2+2? Answer with just the number.")
    if result['success']:
        print(f"      ✅ Response: {result['content'][:100]}")
    else:
        print(f"      ❌ Error: {result['error']}")
        return False
    
    # Test 4: Multi-modal query
    print("\n[4/4] Testing multi-modal query...")
    try:
        # Create a test image (red square)
        test_img = Image.new('RGB', (200, 200), color='red')
        
        result = client.query(
            "What color is this image? Answer in one word.",
            images=[test_img],
            max_tokens=50
        )
        
        if result['success']:
            print(f"      ✅ Response: {result['content'][:100]}")
        else:
            print(f"      ❌ Error: {result['error']}")
            return False
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - Qwen3-VL Client Ready!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    import sys
    success = test_qwen_client()
    sys.exit(0 if success else 1)