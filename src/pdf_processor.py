"""
pdf_processor.py
Extracts text (chunked) and images from a PDF using PyMuPDF (fitz).
"""

import io
import fitz  # PyMuPDF
from PIL import Image
from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    text: str
    page: int
    source_file: str
    chunk_id: str


@dataclass
class ExtractedImage:
    image: Image.Image
    page: int
    source_file: str
    image_id: str


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Simple sliding-window chunker on raw text. Good enough for RAG; swap
    for a smarter splitter later if quality demands it."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start <= 0:
            break
    return [c.strip() for c in chunks if c.strip()]


def extract_text_chunks(pdf_path: str, source_file: str) -> List[TextChunk]:
    doc = fitz.open(pdf_path)
    chunks: List[TextChunk] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")
        page_chunks = chunk_text(raw_text)
        for i, chunk in enumerate(page_chunks):
            chunks.append(
                TextChunk(
                    text=chunk,
                    page=page_num + 1,
                    source_file=source_file,
                    chunk_id=f"{source_file}_p{page_num + 1}_t{i}",
                )
            )
    doc.close()
    return chunks


def extract_images(pdf_path: str, source_file: str, min_size: int = 80) -> List[ExtractedImage]:
    """Pulls every embedded raster image out of the PDF. Skips tiny images
    (icons, bullet decorations) under min_size pixels on either dimension —
    they're noise, not content."""
    doc = fitz.open(pdf_path)
    images: List[ExtractedImage] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        img_list = page.get_images(full=True)
        for img_index, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                continue

            if pil_image.width < min_size or pil_image.height < min_size:
                continue

            images.append(
                ExtractedImage(
                    image=pil_image,
                    page=page_num + 1,
                    source_file=source_file,
                    image_id=f"{source_file}_p{page_num + 1}_img{img_index}",
                )
            )
    doc.close()
    return images
