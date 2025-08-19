# generate_llm.py
import os
import re
import json
from typing import List, Dict, Any, Optional

import requests
from requests.exceptions import ReadTimeout, RequestException

# kita pakai searching.answer_query_auto untuk RAG
import searching as S

# --- detector daftar ---
BULLET_RE = re.compile(r'(?m)^\s*(?:[-•*]|\d+[.)]|\(\d+\)|[A-Za-z][.)])\s+')

def first_list_block(s: str) -> str:
    """Ambil blok daftar (sebelum baris [Sumber])."""
    return re.split(r'\n\s*\[Sumber\]', str(s))[0].strip()

# --- fallback parser sumber dari teks ---
def extract_sources(text: str) -> List[Dict[str, str]]:
    sources, seen = [], set()
    for line in str(text).splitlines():
        m = re.match(r'^\s*\[Sumber\]\s*(.+)$', line.strip())
        if not m:
            continue
        payload = m.group(1)
        if "•" in payload:
            doc, section = [p.strip() for p in payload.split("•", 1)]
        else:
            doc, section = payload.strip(), ""
        key = (doc, section)
        if key not in seen:
            sources.append({"doc": doc, "section": section})
            seen.add(key)
    return sources

def build_sources_from_payload(payload: Any, text: str) -> List[Dict[str, str]]:
    """Gabung sumber terstruktur (payload) + yang ter-parse dari teks."""
    sources, seen = [], set()
    if isinstance(payload, dict) and payload.get("source"):
        s = payload["source"]
        doc_label = s.get("doc_label") or s.get("doc_id", "")
        section   = s.get("section", "")
        key = (doc_label, section)
        if any(doc_label):
            sources.append({"doc": doc_label, "section": section})
            seen.add(key)
    for s in extract_sources(text):
        key = (s["doc"], s["section"])
        if key not in seen:
            sources.append(s)
            seen.add(key)
    return sources

def clean_context(text: str, max_chars: int = 200) -> str:
    """
    Bersihkan konteks: buang baris [Sumber] dan marker bullet/numbering terdepan.
    Potong agar ringkas untuk prompt LLM.
    """
    out = []
    for line in str(text).splitlines():
        if line.strip().startswith("[Sumber]"):
            continue
        line = re.sub(r'^\s*(\d+\.\s*|[-•]\s*)', '', line).strip()
        if line:
            out.append(line)
    cleaned = "\n".join(out).strip()
    return cleaned[:max_chars]

def _default_llm_url() -> str:
    """Ambil URL generator LLM dari ENV, fallback ke localhost."""
    return os.environ.get("LLM_GENERATE_URL", "http://127.0.0.1:8000/generate")

def run_rag_answer(
    query: str,
    top_k_factoid: int = 1,
    llm_url: Optional[str] = None,
    connect_timeout: float = 5.0,
    read_timeout: float = 180.0,
) -> Dict[str, Any]:
    """
    Pipeline: RAG router -> (opsional) ringkas pakai LLM -> hasil final.
    - Jika hasil router berupa daftar (list), langsung passthrough tanpa LLM.
    - Selain itu, LLM dipakai untuk merangkum/mengekstrak jawaban dari konteks.
    """
    # pastikan searching siap (lazy init bila perlu)
    if not S.is_ready():
        # biarkan caller (API) yang memastikan init—di sini fail-fast saja
        raise RuntimeError("Belum init_search().")

    payload = S.answer_query_auto(query, top_k_factoid=top_k_factoid, return_dict=True)
    answer_text = payload["text"] if isinstance(payload, dict) else str(payload)
    answer_type = payload.get("answer_type") if isinstance(payload, dict) else None
    sources = build_sources_from_payload(payload, answer_text)

    # Passthrough bila list
    if answer_type == "list" or BULLET_RE.search(answer_text):
        return {
            "ok": True,
            "answer_type": "list",
            "output": first_list_block(answer_text),
            "finish_reason": "passthrough_list",
            "sumber": sources,
            "used_llm": False,
        }

    # Bersihkan konteks; kalau ternyata masih list, tetap passthrough
    context = clean_context(answer_text)
    if BULLET_RE.search(answer_text):
        return {
            "ok": True,
            "answer_type": answer_type or "list",
            "output": first_list_block(answer_text),
            "finish_reason": "passthrough_list_detected_after_clean",
            "sumber": sources,
            "used_llm": False,
        }

    # Susun prompt LLM
    prompt = (
        "TUGAS: Ekstrak jawaban HANYA dari KONTEN di bawah.\n"
        "- Jika tidak ada info relevan, balas:\n"
        "Tidak ditemukan jawaban pada konteks yang diberikan.\n\n"
        "KONTEN:\n"
        f"\"\"\"{context}\"\"\"\n\n\n"
        "PERTANYAAN:\n"
        f"{query}\n\n"
        "HASIL:"
    )

    llm_url = llm_url or _default_llm_url()

    try:
        resp = requests.post(
            llm_url,
            json={"prompt": prompt},
            timeout=(connect_timeout, read_timeout),
        )
        resp.raise_for_status()
        llm_out = resp.json()

        # normalisasi minimal
        output = llm_out.get("output") or llm_out.get("answer") or llm_out.get("text") or ""
        finish = llm_out.get("finish_reason") or "unknown"

        return {
            "ok": True,
            "answer_type": answer_type or "factoid",
            "output": output.strip() or "Tidak ditemukan jawaban pada konteks yang diberikan.",
            "finish_reason": finish,
            "sumber": sources,
            "used_llm": True,
        }

    except ReadTimeout:
        # fallback aman saat server lambat
        fb = first_list_block(answer_text) if answer_text.strip() else "Tidak ditemukan jawaban pada konteks yang diberikan."
        return {
            "ok": True,
            "answer_type": answer_type or "factoid",
            "output": fb,
            "finish_reason": "client_timeout_fallback",
            "sumber": sources,
            "used_llm": False,
        }
    except RequestException as e:
        return {
            "ok": False,
            "error": f"LLM request error: {e}",
            "answer_type": answer_type or "factoid",
            "output": first_list_block(answer_text) if answer_text.strip() else "",
            "finish_reason": "llm_request_error",
            "sumber": sources,
            "used_llm": False,
        }

# opsional: CLI cepat
# if __name__ == "__main__":
#     q = "daerah mana yang memiliki mobilitas tertinggi?"
#     # NOTE: asumsikan S.init_search() sudah dipanggil sebelumnya
#     print(json.dumps(run_rag_answer(q), ensure_ascii=False, indent=2))
