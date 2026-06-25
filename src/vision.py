"""
vision.py
Uses Groq's vision-capable model (Llama 4 Scout) to turn an extracted image
into a rich text description, so it can be embedded and searched alongside
regular text chunks.
"""

import base64
import io
from groq import Groq
from PIL import Image

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

VISION_PROMPT = (
    "Describe this image in detail for a search index. Include: what type of "
    "visual it is (photo, chart, diagram, table, screenshot, equation, etc.), "
    "all visible text/labels/numbers, and what information or concept it "
    "conveys. Be factual and specific, no fluff. 3-6 sentences."
)


def _pil_to_data_url(image: Image.Image, max_dim: int = 1024) -> str:
    """Downscale large images (keeps Groq requests fast/cheap) and encode as
    a base64 data URL for the chat API."""
    img = image.copy()
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def describe_image(client: Groq, image: Image.Image) -> str:
    """Sends one image to Llama 4 Scout and returns a text description.
    On failure, returns a fallback string rather than crashing the whole
    ingestion pipeline over one bad image."""
    try:
        data_url = _pil_to_data_url(image)
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Image description unavailable: {e}]"
