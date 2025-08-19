# searching.py
# ============================================
# ROUTER & ANSWER HELPERS + LOADER + SEARCH
# ============================================
import re, difflib
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ======== GLOBALS (diinisialisasi via init_search) ========
BASE_DIR = Path(__file__).resolve().parent.parent / "rag_covid"
CLIENT: Optional[PersistentClient] = None
COL = None
DF: Optional[pd.DataFrame] = None
E5: Optional[SentenceTransformer] = None

# (opsional) label dokumen ringkas untuk tampilan
DOC_NAMES = {
    "doc_01-15_171": "Pedoman P2 COVID-19",
    "doc_02-1_10":   "SE No.25/2022 (PPLN)",
    "doc_03-11_22":  "Rencana Operasi COVID-19",
}

# ============================================
# INIT (load Chroma + DataFrame + E5)
# ============================================
def init_search(chroma_path: Optional[Path] = None, collection_name: str = "covid_docs_e5"):
    global CLIENT, COL, DF, E5

    if is_ready():
        return
    with _INIT_LOCK:
        if is_ready():
            return

        if chroma_path is None:
            chroma_path = (BASE_DIR.parent / "chroma_db_e5")

        CLIENT = PersistentClient(path=str(chroma_path))
        COL = CLIENT.get_or_create_collection(name=collection_name)

        total = COL.count()
        got = COL.get(include=["documents", "metadatas"], limit=max(total, 1))
        docs, metas = _normalize_chroma_get(got)

        if not docs or not metas:
            print("[searching.init] WARNING: collection kosong atau tidak ada dokumen/metadata.")
            df = pd.DataFrame(columns=[
                "doc_id","level","tag","title","path","path_full","title_full","content","row_idx","doc"
            ])
        else:
            df = pd.DataFrame(metas)
            df["doc"] = docs
            for col in ["doc_id","level","tag","title","path","path_full","title_full","content"]:
                if col not in df.columns:
                    df[col] = ""
            if "row_idx" not in df.columns:
                df = df.reset_index().rename(columns={"index": "row_idx"})
            if "path_full" not in df.columns:
                df["path_full"] = np.where(
                    df["path"].astype(str) == "ROOT",
                    df["title"].astype(str),
                    df["path"].astype(str) + " > " + df["title"].astype(str),
                )

        # simpan ke global DF di AKHIR supaya is_ready() akurat
        DF = df

        if E5 is None:
            E5 = SentenceTransformer("intfloat/multilingual-e5-base")

        print(f"[searching.init] using collection: {collection_name} count: {total}")


# ============================================
# UTIL PARSER & FORMAT
# ============================================
def _looks_like_sentence(seg: str) -> bool:
    seg = (seg or "").strip()
    if not seg: return False
    if len(seg) > 60: return True
    if len(seg.split()) > 12: return True
    if seg.endswith("."): return True
    if "," in seg: return True
    return False

def compact_head_from_parts(parts, keep_last=2, max_chars=120):
    parts = [p.strip() for p in parts if p.strip()]
    if not parts: return ""
    if len(parts) >= 2 and _looks_like_sentence(parts[-1]):
        parts = parts[:-1]
    if len(parts) > keep_last:
        parts = parts[-keep_last:]
    head_short = " > ".join(parts)
    head_short = re.sub(r"\s+", " ", head_short).strip()
    return head_short if len(head_short) <= max_chars else head_short[:max_chars-1] + "…"

def compact_head(head_str: str, keep_last=2, max_chars=120):
    parts = [p.strip() for p in str(head_str).split(">")]
    return compact_head_from_parts(parts, keep_last=keep_last, max_chars=max_chars)

def fmt_source(meta: dict, head_fallback: str = ""):
    doc_label = DOC_NAMES.get(meta.get("doc_id",""), meta.get("doc_id","(unknown)"))
    if meta.get("path") is not None and meta.get("title") is not None:
        parts = ([meta["path"]] if meta["path"] != "ROOT" else []) + [meta["title"]]
        head_disp = compact_head_from_parts(parts, keep_last=2, max_chars=120)
    else:
        head_disp = compact_head(head_fallback, keep_last=2, max_chars=120)
    return f"[Sumber] {doc_label} • {head_disp}"

ITEM_PAT = re.compile(r'^\s*(\d+\.|[a-z]\.|•|-|\(\d+\)|\([a-z]\))', re.IGNORECASE)

def strip_leading_markers(text: str) -> str:
    t = str(text or "").strip()
    if not t: return ""
    for _ in range(5):
        new_t = re.sub(
            r'^\s*((\d+(\.\d+)*)|[ivxlcdm]+|[a-z]|[-•]|\(\d+\)|\([a-z]\))(\.|\)|:)?\s+',
            '', t, flags=re.I
        )
        if new_t == t: break
        t = new_t.strip()
    return t

def first_sentence(text, max_chars=220):
    t = strip_leading_markers(text)
    parts = re.split(r'(?<=[.!?])\s+|\n+', t)
    out = ""
    p_idx = 0
    for i, p in enumerate(parts):
        s = p.strip()
        if len(s) < 6 or not re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', s): continue
        out = s; p_idx = i; break
    if out:
        j = p_idx + 1
        while len(out) < 80 and j < len(parts):
            nxt = parts[j].strip()
            if len(nxt) >= 6 and re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', nxt):
                out = (out + " " + nxt).strip()
            j += 1
    else:
        out = t[:max_chars].strip()
    return out[:max_chars]

def display_head_from_meta(m):
    return m["title_full"] if m.get("path")=="ROOT" else f"{m.get('path_full','')} > {m.get('title_full','')}"

def head_key_from_meta(m):
    return m["title"] if m.get("path")=="ROOT" else f"{m.get('path','')} > {m.get('title','')}"

def split_head_content(doc_str):
    parts = str(doc_str).split("||", 1)
    head = parts[0].strip()
    content = parts[1].strip() if len(parts) > 1 else ""
    return head, content

def _tok(s):  # tokenisasi ringan
    return re.findall(r"\w+", (s or "").lower())

def _soft_overlap(q_tokens, s_tokens):
    score = 0.0
    s_tokens = list(s_tokens)
    sset = set(s_tokens)
    for qt in q_tokens:
        if qt in sset:
            score += 1.0
            continue
        pref = qt[: max(2, int(len(qt)*0.5))]
        if any(st.startswith(pref) for st in s_tokens):
            score += 0.5
            continue
        best = max((difflib.SequenceMatcher(None, qt, st).ratio() for st in s_tokens), default=0.0)
        if best >= 0.8:
            score += 0.5
    return score


# ============================================
# DF-DEPENDENT HELPERS
# ============================================
def ensure_df_ready():
    if DF is None:
        raise RuntimeError("DF belum diinisialisasi. Panggil init_search() lebih dulu.")

def get_df() -> pd.DataFrame:
    ensure_df_ready()
    return DF

def _ensure_row_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "row_idx" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_idx"})
    if "full_path" not in df.columns:
        df["full_path"] = np.where(df["path"]=="ROOT", df["title"], df["path"] + " > " + df["title"])
    return df

def get_children_rows(doc_id, parent_full_path, include_descendants=False, group_to_child=True):
    df = _ensure_row_cols(get_df())
    if include_descendants:
        mask = (df["doc_id"] == doc_id) & (df["path"].str.startswith(parent_full_path))
    else:
        mask = (df["doc_id"] == doc_id) & (df["path"] == parent_full_path)

    rows = df.loc[mask].copy()
    if rows.empty:
        cols = ["doc_id","level","tag","title","path","full_path","content","row_idx"]
        return pd.DataFrame(columns=[c for c in cols if c in df.columns])

    # group per anak (jika data per-kalimat)
    IS_SENT_MODE = ("sent_idx" in rows.columns) and rows["sent_idx"].notna().any()
    if IS_SENT_MODE and group_to_child:
        rows = rows.sort_values("row_idx")
        keys = ["doc_id","path","title","level","tag"]
        g = rows.groupby(keys, as_index=False).agg({
            "content": lambda s: " ".join([str(x).strip() for x in s if str(x).strip()]),
            "row_idx": "min",
        })
        g["full_path"] = np.where(g["path"]=="ROOT", g["title"], g["path"]+" > "+g["title"])
        return g[["doc_id","level","tag","title","path","full_path","content","row_idx"]].sort_values("row_idx")

    return rows.sort_values("row_idx")[["doc_id","level","tag","title","path","full_path","content","row_idx"]]

def children_stats_for_doc(doc_id):
    df = get_df()
    d = df[df["doc_id"] == doc_id]
    if d.empty:
        return {"median": 0, "p75": 0}
    uniq_child = d[["path","title","level"]].drop_duplicates()
    counts = uniq_child.groupby("path").size()
    if counts.empty:
        return {"median": 0, "p75": 0}
    return {"median": int(np.median(counts.values)), "p75": int(np.percentile(counts.values, 75))}

def structure_scores(doc_id, parent_full_path):
    rows = get_children_rows(doc_id, parent_full_path, include_descendants=False, group_to_child=True)
    child_count = len(rows)
    if child_count == 0:
        return {"child_count":0, "item_ratio":0.0, "descendant_count":0, "avg_child_len":0}
    titles = rows["title"].astype(str).tolist()
    items = [t for t in titles if ITEM_PAT.search(t.strip())]
    item_ratio = (len(items) / child_count) if child_count else 0.0
    desc_rows = get_children_rows(doc_id, parent_full_path, include_descendants=True, group_to_child=True)
    descendant_count = max(0, len(desc_rows) - child_count)
    texts = rows["content"].astype(str).tolist()
    avg_child_len = int(np.mean([len(t) for t in texts])) if texts else 0
    return {
        "child_count": child_count,
        "item_ratio": item_ratio,
        "descendant_count": descendant_count,
        "avg_child_len": avg_child_len
    }


# ============================================
# SEARCH (pakai Chroma + E5)
# ============================================
def search(query: str, n_results: int = 30) -> List[Dict[str, Any]]:
    if COL is None or E5 is None:
        raise RuntimeError("Belum init_search().")
    q_emb = E5.encode(["query: " + query], normalize_embeddings=True).tolist()
    res = COL.query(query_embeddings=q_emb, n_results=n_results, include=["documents","metadatas","distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if "distances" in res else [0.0]*len(docs)

    q_tokens = _tok(query)
    hits: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        head, content = split_head_content(doc)
        s_tokens = _tok(head + " " + content)
        overlap = _soft_overlap(q_tokens, s_tokens)
        hits.append({
            "doc": doc,
            "meta": meta,
            "dist": float(dist),
            "overlap": float(overlap),
        })
    return hits


# ============================================
# ROUTER & ANSWER
# ============================================
def choose_parent(hits, max_level_bonus=3):
    best, best_score = None, -1e9
    for h in hits:
        m = h["meta"]
        lvl_str = str(m.get("level","99"))
        lvl = int(lvl_str) if lvl_str.isdigit() else 99
        score = -h["dist"] + 0.02*h["overlap"]
        if lvl <= max_level_bonus:
            score += 0.3
        if score > best_score:
            best_score = score
            best = h
    return best

def answer_list(parent_meta, include_descendants=False, max_items=None):
    head_display = display_head_from_meta(parent_meta)
    head_key     = head_key_from_meta(parent_meta)
    rows = get_children_rows(parent_meta["doc_id"], head_key, include_descendants=include_descendants, group_to_child=True)
    if rows.empty:
        return f"Tidak ada butir di bawah **{head_display}**.\n\n{fmt_source(parent_meta, head_display)}"

    titles = rows["title"].astype(str).tolist()
    is_item = [bool(ITEM_PAT.search(t.strip())) for t in titles]

    out_lines = []
    for i, (_, r) in enumerate(rows.iterrows()):
        ttl = (r["title"] or "").strip()
        cnt = r.get("content","") or ""
        sent = first_sentence(cnt)
        if is_item[i]:
            if not sent or _norm(ttl) == _norm(sent) or _norm(ttl) in _norm(sent) or _norm(sent) in _norm(ttl):
                line = f"- **{ttl.rstrip(':;')}**"
            else:
                line = f"- **{ttl.rstrip(':;')}** — {sent}"
        else:
            line = f"- {sent}" if sent else f"- {ttl}"
        out_lines.append(line)

    if max_items:
        out_lines = out_lines[:max_items]
    body = "\n".join(out_lines)
    return f"Berikut daftar di bawah **{head_display}**:\n{body}\n\n{fmt_source(parent_meta, head_display)}"

def answer_chapter_summary(parent_hit, max_children=12):
    m = parent_hit["meta"]
    head_display = display_head_from_meta(m)
    head_key     = head_key_from_meta(m)
    children = get_children_rows(m["doc_id"], head_key, include_descendants=False, group_to_child=True)
    if children.empty:
        _, parent_content = split_head_content(parent_hit["doc"])
        snippet = first_sentence(parent_content, 300)
        return f"{snippet}\n\n{fmt_source(m, head_display)}"
    bullets = []
    for _, r in children.sort_values("row_idx").head(max_children).iterrows():
        ttl_full = r.get("title_full", r["title"])
        bullets.append(f"- **{ttl_full}** — {first_sentence(r['content'])}")
    return f"Gambaran umum **{head_display}**:\n" + "\n".join(bullets) + f"\n\n{fmt_source(m, head_display)}"

def _norm(s):
    return strip_leading_markers(str(s or "")).strip().rstrip(":;").lower()

def best_sentence_for_query(text, query, fallback_chars=400):
    t = strip_leading_markers(text)
    cands = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', t) if s.strip()]
    if not cands: return t[:fallback_chars]
    q = _tok(query)
    best, best_score = cands[0], -1e9
    for s in cands:
        st = _tok(s)
        exact = len(set(q) & set(st))
        fuzzy = _soft_overlap(q, st)
        score = 1.0*exact + 0.6*fuzzy + 0.002*len(s)
        if score > best_score:
            best, best_score = s, score
    idx = cands.index(best)
    if len(best) < 80 and idx+1 < len(cands):
        nxt = cands[idx+1]
        if len(nxt) > 20:
            best = (best + " " + nxt).strip()
    return best[:fallback_chars]

def answer_factoid(hits, query_text=None, top_k=1, diversify=True):
    scored = []
    for h in hits:
        head_str, content = split_head_content(h["doc"])
        lvl_str = str(h["meta"].get("level","99"))
        lvl = int(lvl_str) if lvl_str.isdigit() else 99
        score = -h["dist"] + 0.02*h["overlap"]
        if lvl <= 3: score += 0.25
        scored.append((score, h, head_str, content))
    if not scored:
        return "Tidak ditemukan jawaban yang cukup spesifik."
    scored.sort(key=lambda x: x[0], reverse=True)

    picked, seen = [], set()
    for _, h, head_str, content in scored:
        key = (h["meta"]["doc_id"], head_str) if diversify else id(h)
        if key in seen: continue
        picked.append((h, head_str, content)); seen.add(key)
        if len(picked) >= max(1, int(top_k)): break

    if len(picked) == 1:
        h, head_str, content = picked[0]
        sent = best_sentence_for_query(content, query_text or "")
        head_display = display_head_from_meta(h["meta"])
        return f"{sent}\n\n{fmt_source(h['meta'], head_display)}"

    lines = []
    for i, (h, head_str, content) in enumerate(picked, 1):
        sent = best_sentence_for_query(content, query_text or "")
        head_display = display_head_from_meta(h["meta"])
        lines.append(f"{i}. {sent}\n   {fmt_source(h['meta'], head_display)}")
    return "\n\n".join(lines)

def answer_query_auto(query, top_k_factoid=1, return_dict=False):
    hits = search(query, n_results=80)
    if not hits:
        payload = {'answer_type': 'none', 'text': "Maaf, tidak ada hasil relevan.", 'source': {}}
        return payload if return_dict else payload['text']

    parent = choose_parent(hits)

    def _src(meta, section_full_path):
        return {
            'doc_id':   meta.get('doc_id', ''),
            'doc_label': DOC_NAMES.get(meta.get('doc_id', ''), meta.get('doc_id', '')),
            'section':  compact_head(section_full_path, keep_last=2),
        }

    if not parent:
        txt = answer_factoid(hits, query_text=query, top_k=top_k_factoid)
        meta0 = hits[0]['meta']; head0, _ = split_head_content(hits[0]['doc'])
        payload = {'answer_type': 'factoid', 'text': txt, 'source': _src(meta0, head0)}
        return payload if return_dict else txt

    meta = parent["meta"]
    parent_full_path = meta["title"] if meta.get("path")=="ROOT" else f"{meta.get('path','')} > {meta.get('title','')}"
    lvl_str = str(meta.get("level","99"))
    lvl = int(lvl_str) if lvl_str.isdigit() else 99

    gstats = children_stats_for_doc(meta["doc_id"])
    s = structure_scores(meta["doc_id"], parent_full_path)

    list_condition = (s["child_count"] >= max(3, gstats["median"])) and (s["item_ratio"] >= 0.45)
    chapter_condition = (
        s["child_count"] >= max(4, gstats["p75"]) or
        (s["child_count"] >= 3 and s["descendant_count"] >= s["child_count"] * 0.3) or
        lvl <= 2
    )

    if list_condition and not chapter_condition:
        txt = answer_list(meta, include_descendants=False)
        payload = {'answer_type': 'list', 'text': txt, 'source': _src(meta, parent_full_path)}
        return payload if return_dict else txt

    if chapter_condition and not list_condition:
        txt = answer_chapter_summary(parent, max_children=12)
        payload = {'answer_type': 'chapter_summary', 'text': txt, 'source': _src(meta, parent_full_path)}
        return payload if return_dict else txt

    if list_condition and chapter_condition:
        if s["item_ratio"] >= 0.6 or s["avg_child_len"] <= 140:
            txt = answer_list(meta, include_descendants=False)
            payload = {'answer_type': 'list', 'text': txt, 'source': _src(meta, parent_full_path)}
        elif s["descendant_count"] >= s["child_count"] * 0.5:
            txt = answer_chapter_summary(parent, max_children=12)
            payload = {'answer_type': 'chapter_summary', 'text': txt, 'source': _src(meta, parent_full_path)}
        else:
            txt = answer_chapter_summary(parent, max_children=12)
            payload = {'answer_type': 'chapter_summary', 'text': txt, 'source': _src(meta, parent_full_path)}
        return payload if return_dict else payload['text']

    txt = answer_factoid(hits, query_text=query, top_k=top_k_factoid)
    meta0 = hits[0]['meta']; head0, _ = split_head_content(hits[0]['doc'])
    payload = {'answer_type': 'factoid', 'text': txt, 'source': _src(meta0, head0)}
    return payload if return_dict else txt

import threading

CLIENT = None
COL = None
DF = None
E5 = None
_INIT_LOCK = threading.Lock()

def is_ready() -> bool:
    return (CLIENT is not None) and (COL is not None) and (DF is not None) and (E5 is not None)

def get_df():
    if DF is None:
        raise RuntimeError("Belum init_search()")
    return DF


def _normalize_chroma_get(payload):
    """
    Normalisasi hasil COL.get(...) agar selalu:
      docs: List[str], metas: List[dict]
    Menangani variasi: list-of-list, single string, single dict, kosong.
    """
    docs = payload.get("documents", [])
    metas = payload.get("metadatas", [])

    # unwrap nested list [[...]] -> [...]
    if isinstance(docs, list) and docs and isinstance(docs[0], list):
        docs = docs[0]
    if isinstance(metas, list) and metas and isinstance(metas[0], list):
        metas = metas[0]

    # singletons -> list
    if isinstance(docs, str):
        docs = [docs]
    if isinstance(metas, dict):
        metas = [metas]

    # fallback kosong
    docs = docs or []
    metas = metas or []

    # samakan panjang bila tak selaras (ambil minimum aman)
    if len(docs) != len(metas):
        n = min(len(docs), len(metas))
        docs = docs[:n]
        metas = metas[:n]

    return docs, metas
