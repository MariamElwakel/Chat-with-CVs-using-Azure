import uuid
import fitz
import pymupdf4llm

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
from rank_bm25 import BM25Okapi

from config import (
    qdrant_client,
    embeddings,
    COLLECTION_NAME,
    EMBEDDING_DIM,
)

from experience import calculate_years_of_experience


def chunking_CVs(uploaded_files):
    """
    - Converts each PDF to markdown
    - Splits content by headers (H1, H2, H3)
    - Extracts candidate name from the first line
    - Computes total years of experience once per CV
    - Attaches metadata (candidate info + headers + experience) to each chunk

    Returns a list of Document chunks.
    """

    chunks = []

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for file in uploaded_files:

        file_id = str(uuid.uuid4())

        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        md_text = pymupdf4llm.to_markdown(doc)

        # Extract candidate name from first markdown line
        lines = [l.strip() for l in md_text.split("\n") if l.strip()]
        first_line = lines[0].replace("#", "").strip()

        md_docs = splitter.split_text(md_text)

        # Calculate years of experience once per CV from experience sections only
        years = calculate_years_of_experience(md_docs)

        for d in md_docs:
            content = f"Candidate: {first_line}\n\n{d.page_content}"

            meta = {
                "cv_id": file_id,
                "candidate_name": first_line,
                "file_name": file.name,
                **d.metadata
            }

            # Attach years_of_experience to every chunk of this CV if found
            if years is not None:
                meta["years_of_experience"] = years

            chunks.append(
                Document(
                    page_content=content,
                    metadata=meta
                )
            )

    return chunks



def build_bm25_index(chunks):
    """
    Build a BM25 keyword-based search index from the chunks.
    """
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25


def build_vectorstore(uploaded_files):
    """
    Create a Qdrant vector database for CV search.

    - Resets (deletes + recreates) the collection
    - Defines vector configuration (embedding size + cosine similarity)
    - Adds payload index for filtering (e.g., years_of_experience)
    - Chunks CVs and computes metadata
    - Builds BM25 index for keyword search
    - Embeds and uploads chunks to Qdrant
    """
    
    # Delete old collection if it exists
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)

    # Create fresh collection
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

    # Create a payload index so Qdrant can filter on years_of_experience.
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="years_of_experience",
        field_schema=PayloadSchemaType.FLOAT,
    )

    # Chunk all CVs
    chunks = chunking_CVs(uploaded_files)

    # Build BM25 index over the same chunks for keyword search
    bm25_index = build_bm25_index(chunks)

    # Embed and upload chunks to Qdrant
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    vectorstore.add_documents(chunks, batch_size=16)

    return vectorstore, bm25_index, chunks, len(chunks)