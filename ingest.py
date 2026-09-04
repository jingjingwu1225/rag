"""
ingest.py
Run this ONCE (and again any time you add/change a PDF in docs/) to build
the knowledge base:

    python ingest.py

Steps: read every PDF in docs/ -> extract text -> chunk it -> embed each
chunk -> store it in the local Chroma vector database (chroma_db/).
"""

import glob
import os

from pypdf import PdfReader

from rag_core import chunk_text, embed_texts, get_collection, strip_boilerplate

DOCS_DIR = "docs"
BATCH_SIZE = 64  # how many chunks to embed per API call


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def main():
    pdf_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {DOCS_DIR}/. Add your papers there first.")
        return

    collection = get_collection()

    all_chunks, all_metadatas, all_ids = [], [], []

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"Reading {filename} ...")

        # Clear out this file's old chunks first. Chunk boundaries (and
        # counts) can change between runs — e.g. if chunk_text() changes,
        # or the PDF is updated — and upsert() only overwrites ids that
        # still exist; it won't clean up stale ids left over from a
        # previous, larger chunking of the same file. Without this,
        # re-running ingest.py can leave orphaned old chunks in Chroma
        # that never show up in ingest.py's own count but still get
        # retrieved at query time.
        collection.delete(where={"source": filename})

        text = strip_boilerplate(extract_text_from_pdf(pdf_path))
        chunks = chunk_text(text)
        print(f"  -> {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": filename, "chunk_index": i})
            all_ids.append(f"{filename}::{i}")

    print(f"\nEmbedding {len(all_chunks)} chunks total (in batches of {BATCH_SIZE})...")
    for start in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[start:start + BATCH_SIZE]
        batch_metadatas = all_metadatas[start:start + BATCH_SIZE]
        batch_ids = all_ids[start:start + BATCH_SIZE]

        batch_embeddings = embed_texts(batch_chunks)

        collection.upsert(
            documents=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )
        print(f"  embedded + stored chunks {start} - {start + len(batch_chunks)}")

    print("\nDone. Knowledge base is ready — try `python query.py` or `streamlit run app.py`.")


if __name__ == "__main__":
    main()
