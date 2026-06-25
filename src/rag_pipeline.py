"""
rag_pipeline.py
Orchestrates the whole flow:
  ingest:  PDF -> text chunks + images -> image captions -> embeddings -> FAISS
  query:   question -> embedding -> FAISS retrieve -> Groq chat completion
"""

from groq import Groq
from src.pdf_processor import extract_text_chunks, extract_images
from src.vision import describe_image
from src.embeddings import Embedder
from src.vector_store import VectorStore

CHAT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about uploaded PDF "
    "documents. You are given retrieved excerpts from the documents — some "
    "are raw text, some are descriptions of images/charts/diagrams found in "
    "the PDF. Answer ONLY using the provided context. If the context doesn't "
    "contain the answer, say so plainly — do not make things up. When you "
    "use an image-derived excerpt, mention that the information came from "
    "an image/figure. Always cite the source file and page number for "
    "claims you make, like (source.pdf, page 4)."
)


class RAGPipeline:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.embedder = Embedder()
        self.store = VectorStore(dim=self.embedder.dim)

    def ingest_pdf(self, pdf_path: str, source_file: str, progress_callback=None):
        """Extracts text + images, captions images via vision model, embeds
        everything, and adds it to the vector store. progress_callback(str)
        is optional, for UI status updates."""

        def report(msg):
            if progress_callback:
                progress_callback(msg)

        report(f"Extracting text from {source_file}...")
        text_chunks = extract_text_chunks(pdf_path, source_file)

        report(f"Extracting images from {source_file}...")
        images = extract_images(pdf_path, source_file)

        # --- Text ---
        if text_chunks:
            report(f"Embedding {len(text_chunks)} text chunks...")
            texts = [c.text for c in text_chunks]
            vectors = self.embedder.embed(texts)
            metadatas = [
                {
                    "type": "text",
                    "content": c.text,
                    "page": c.page,
                    "source_file": c.source_file,
                    "id": c.chunk_id,
                }
                for c in text_chunks
            ]
            self.store.add(vectors, metadatas)

        # --- Images ---
        if images:
            report(f"Describing {len(images)} images with vision model...")
            captions = []
            for i, img in enumerate(images):
                report(f"  - describing image {i + 1}/{len(images)}...")
                caption = describe_image(self.client, img.image)
                captions.append(caption)

            report(f"Embedding {len(images)} image captions...")
            vectors = self.embedder.embed(captions)
            metadatas = [
                {
                    "type": "image",
                    "content": caption,
                    "page": img.page,
                    "source_file": img.source_file,
                    "id": img.image_id,
                }
                for img, caption in zip(images, captions)
            ]
            self.store.add(vectors, metadatas)

        report(f"Done with {source_file}: {len(text_chunks)} text chunks, {len(images)} images.")
        return len(text_chunks), len(images)

    def save(self, path: str):
        self.store.save(path)

    def load(self, path: str):
        self.store = VectorStore.load(path, self.embedder.dim)

    def retrieve(self, query: str, k: int = 6):
        query_vector = self.embedder.embed([query])[0]
        return self.store.search(query_vector, k=k)

    def answer(self, query: str, k: int = 6) -> dict:
        results = self.retrieve(query, k=k)

        if not results:
            return {
                "answer": "No documents have been ingested yet, so I have nothing to search.",
                "sources": [],
            }

        context_blocks = []
        for r in results:
            tag = "IMAGE DESCRIPTION" if r["type"] == "image" else "TEXT"
            context_blocks.append(
                f"[{tag} | {r['source_file']} | page {r['page']}]\n{r['content']}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        user_prompt = (
            f"Context excerpts from the documents:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer the question using only the context above."
        )

        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": results,
        }
