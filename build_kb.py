# build_py/build_kb.py
# ============================================
# IMPORTS
# ============================================
import os, json, re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ============================================
# PATH & KONFIG
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent / "rag_covid"

# Default file JSON (boleh override via argumen run_build_kb)
DEFAULT_FILE_PATHS = [
    BASE_DIR / "dataset" / "doc_01-15_171.json",
    BASE_DIR / "dataset" / "doc_02-1_10.json",
    BASE_DIR / "dataset" / "doc_03-11_22.json",
]

DOC_TITLES: Dict[str, str] = {
    "doc_01-15_171": "Pedoman Pencegahan & Pengendalian COVID-19",
    "doc_03-11_22":  "Rencana Operasi Penanggulangan COVID-19 (Bidang Kesehatan)",
    "doc_02-1_10":   "Surat Edaran No. 25/2022: Protokol Kesehatan PPLN",
}

# ============================================
# HELPERS
# ============================================
ITEM_PAT = re.compile(r'^\s*(\d+\.|[a-z]\.|•|-|\(\d+\)|\([a-z]\))\s*', re.IGNORECASE)

def short_title(text: str, max_words: int = 12) -> str:
    words = str(text or "").strip().split()
    return " ".join(words[:max_words])

def strip_leading_markers(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    for _ in range(5):
        new_t = re.sub(
            r'^\s*((\d+(\.\d+)*)|[ivxlcdm]+|[a-z]|[-•]|\(\d+\)|\([a-z]\))(\.|\)|:)?\s+',
            '',
            t,
            flags=re.I
        )
        if new_t == t:
            break
        t = new_t.strip()
    return t

def split_into_sentences(text: str) -> List[str]:
    """
    Split ringan untuk Indonesia:
    - pecah di akhir kalimat [.?!]
    - hormati newline (sering berarti pergantian poin/list)
    - bersihkan marker penomoran di awal segmen
    """
    if not text:
        return []
    t = re.sub(r'\r\n?', '\n', text).strip()
    t = re.sub(r'\n\s*' + ITEM_PAT.pattern, '\n', t)  # bullet di awal baris → pemisah
    parts = re.split(r'(?<=[.!?])\s+|\n+', t)
    sents: List[str] = []
    for p in parts:
        p = strip_leading_markers(p.strip())
        if not p:
            continue
        if len(p) < 6 or not re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', p):
            continue
        sents.append(p)
    return sents or [text.strip()]

def flatten_nodes(
    nodes: List[Dict[str, Any]],
    doc_id: str,
    parent_titles_short: Optional[List[str]] = None,
    parent_titles_full: Optional[List[str]] = None,
    rows: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    if rows is None:
        rows = []
    if parent_titles_short is None:
        parent_titles_short = []
    if parent_titles_full is None:
        parent_titles_full = []

    for node in nodes:
        txt = (node.get("text") or "").strip()
        if not txt:
            continue
        lvl = node.get("level", "")
        tag = node.get("tag", "")

        title_full  = short_title(txt, 40)   # untuk display
        title_short = short_title(txt, 12)   # untuk path kunci
        path_short  = " > ".join(parent_titles_short) or "ROOT"
        path_full   = " > ".join(parent_titles_full)  or "ROOT"

        rows.append({
            "doc_id": doc_id,
            "level": lvl,
            "tag": tag,
            "title": title_short,       # kunci path
            "path":  path_short,        # kunci path
            "title_full": title_full,   # display
            "path_full":  path_full,    # display
            "content": txt,
            "sent_idx": None,
        })

        children = node.get("children", [])
        if children:
            flatten_nodes(
                children, doc_id,
                parent_titles_short = parent_titles_short + [title_short],
                parent_titles_full  = parent_titles_full  + [title_full],
                rows=rows
            )
    return rows

def build_doc(row: pd.Series) -> str:
    head_path = row["title_full"] if row["path"] == "ROOT" else f"{row['path']} > {row['title_full']}"
    head = f"[{row['doc_title']}] {head_path}"
    return f"{head} || {row['content']}"

def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    is_query: bool = False,
    batch_size: int = 32,
    show_progress: bool = False
):
    prefix = "query: " if is_query else "passage: "
    return model.encode(
        [prefix + t for t in texts],
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        batch_size=batch_size,
    )

# ============================================
# ENTRYPOINT YANG DIPANGGIL DARI FASTAPI
# ============================================
def run_build_kb(
    file_paths: Optional[List[Path]] = None,
    chroma_path: Optional[Path] = None,
    collection_name: str = "covid_docs_e5",
    embed_batch_size: int = 32,
    show_embed_progress: bool = False
) -> Dict[str, Any]:
    """
    Build KB: flatten JSON → buat dokumen → embed (E5) → simpan ke Chroma.

    Returns:
        dict ringkasan hasil (status, jumlah chunk, collection, path db).
    """
    file_paths = file_paths or DEFAULT_FILE_PATHS
    chroma_path = chroma_path or Path(os.environ.get("CHROMA_PATH", BASE_DIR / "chroma_db_e5"))

    # 1) Baca & flatten
    all_rows: List[Dict[str, Any]] = []
    for p in file_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc_id = os.path.splitext(os.path.basename(p))[0]
        all_rows.extend(flatten_nodes(data, doc_id))

    df = pd.DataFrame(all_rows)
    if df.empty:
        return {"status": "empty", "chunks": 0, "collection": collection_name, "db_path": str(chroma_path)}

    df["doc_title"] = df["doc_id"].map(DOC_TITLES).fillna(df["doc_id"])
    df["doc"] = df.apply(build_doc, axis=1)

    # 2) Embedding (E5 multilingual)
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    documents: List[str] = df["doc"].tolist()
    embeddings = embed_texts(
        model, documents,
        is_query=False,
        batch_size=embed_batch_size,
        show_progress=show_embed_progress
    ).tolist()

    # 3) Simpan ke Chroma
    client = PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(name=collection_name)

    ids = [str(i) for i in range(len(df))]
    metadatas = [{
        "doc_id":     str(r["doc_id"]),
        "doc_title":  str(r["doc_title"]),
        "level":      str(r["level"]),
        "path":       str(r["path"]),
        "title":      str(r["title"]),
        "path_full":  str(r["path_full"]),
        "title_full": str(r["title_full"]),
    } for _, r in df.iterrows()]

    col.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    return {
        "status": "success",
        "chunks": int(col.count()),
        "collection": collection_name,
        "db_path": str(chroma_path),
    }

# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    result = run_build_kb()
    # print(result)
