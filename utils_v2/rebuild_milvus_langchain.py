from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path
from typing import Iterator

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections, utility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings


def _connect_milvus() -> None:
    settings = get_settings()
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
        db_name=settings.milvus_db_name,
        timeout=settings.milvus_timeout,
        secure=settings.milvus_secure,
        user=settings.milvus_user or None,
        password=settings.milvus_password or None,
    )


def _pick_and_copy_pdfs(source_dir: Path, selected_dir: Path, max_files: int) -> list[Path]:
    selected_dir.mkdir(parents=True, exist_ok=True)
    for old in selected_dir.glob("*.pdf"):
        old.unlink(missing_ok=True)
    pdf_files = sorted(source_dir.rglob("*.pdf"))
    picked = pdf_files[:max_files]
    copied: list[Path] = []
    for idx, src in enumerate(picked, start=1):
        dst = selected_dir / f"{idx:03d}_{src.name}"
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _load_pdf_text(path: Path) -> str:
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    return "\n".join(page.page_content for page in pages if page.page_content.strip())


def _iter_parent_child_rows(pdf_files: list[Path]) -> Iterator[tuple[str, str, dict]]:
    settings = get_settings()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.v2_parent_chunk_size,
        chunk_overlap=settings.v2_parent_chunk_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.v2_child_chunk_size,
        chunk_overlap=settings.v2_child_chunk_overlap,
    )
    total_files = len(pdf_files)
    for file_idx, doc in enumerate(pdf_files, start=1):
        print(f"[parse] ({file_idx}/{total_files}) loading: {doc.name}")
        text = _load_pdf_text(doc)
        if not text.strip():
            print(f"[parse] ({file_idx}/{total_files}) skip empty: {doc.name}")
            continue
        source = str(doc)
        title = doc.stem
        parent_docs = parent_splitter.create_documents([text], metadatas=[{"source": source, "title": title}])
        produced_children = 0
        for parent_idx, parent in enumerate(parent_docs):
            parent_id = f"{doc.stem}_{parent_idx}_{uuid.uuid4().hex[:8]}"
            children = child_splitter.create_documents([parent.page_content], metadatas=[parent.metadata])
            for child_idx, child in enumerate(children):
                child_id = f"{parent_id}_c{child_idx}"
                metadata = {
                    "parent_text": parent.page_content,
                    "parent_id": parent_id,
                    "source": source,
                    "title": title,
                    "chunk_index": child_idx,
                }
                produced_children += 1
                yield (child_id, child.page_content, metadata)
        print(
            f"[chunk] ({file_idx}/{total_files}) parent_chunks={len(parent_docs)} child_chunks={produced_children} file={doc.name}"
        )


def _flush_batch(
    vectorstore: Milvus,
    batch_rows: list[tuple[str, str, dict]],
    *,
    inserted_so_far: int,
) -> int:
    ids = [row[0] for row in batch_rows]
    texts = [row[1] for row in batch_rows]
    metadatas = [row[2] for row in batch_rows]
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    total = inserted_so_far + len(batch_rows)
    print(f"[insert] batch={len(batch_rows)} inserted_total={total}")
    return total


def rebuild_collection(pdf_files: list[Path], *, drop_existing: bool = True, batch_size: int = 256) -> None:
    settings = get_settings()

    _connect_milvus()
    if drop_existing and utility.has_collection(settings.v2_milvus_collection):
        utility.drop_collection(settings.v2_milvus_collection)
        print(f"Dropped existing collection: {settings.v2_milvus_collection}")

    embedding = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name=settings.v2_milvus_collection,
        connection_args={
            "uri": uri,
            "user": settings.milvus_user or None,
            "password": settings.milvus_password or None,
            "db_name": settings.milvus_db_name,
        },
        auto_id=False,
        text_field="child_text",
        primary_field="child_id",
        vector_field="embedding",
    )
    print(f"[milvus] target_collection={settings.v2_milvus_collection}")
    print(f"[milvus] insert_mode=batch batch_size={batch_size}")

    inserted_total = 0
    batch_rows: list[tuple[str, str, dict]] = []
    for row in _iter_parent_child_rows(pdf_files):
        batch_rows.append(row)
        if len(batch_rows) >= batch_size:
            inserted_total = _flush_batch(vectorstore, batch_rows, inserted_so_far=inserted_total)
            batch_rows.clear()
    if batch_rows:
        inserted_total = _flush_batch(vectorstore, batch_rows, inserted_so_far=inserted_total)
        batch_rows.clear()

    if inserted_total == 0:
        print("No PDF chunks were produced.")
        return
    print(f"[done] rebuilt collection {settings.v2_milvus_collection} with {inserted_total} child chunks.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select PDF files and rebuild v2 Milvus with parent/child chunks.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("source_data/raw_data"),
        help="Directory containing raw files (PDFs will be selected).",
    )
    parser.add_argument(
        "--selected-pdf-dir",
        type=Path,
        default=Path("source_data/selected_pdf_100"),
        help="Destination folder to store selected PDF files.",
    )
    parser.add_argument(
        "--max-pdf",
        type=int,
        default=100,
        help="Number of PDF files to select from source dir.",
    )
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="Keep existing collection and append data instead of dropping it first.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Milvus insert batch size for child chunks.",
    )
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    selected_dir = args.selected_pdf_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    selected_pdfs = _pick_and_copy_pdfs(source_dir, selected_dir, args.max_pdf)
    print(f"Selected {len(selected_pdfs)} PDFs into: {selected_dir}")
    rebuild_collection(selected_pdfs, drop_existing=not args.keep_collection, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
