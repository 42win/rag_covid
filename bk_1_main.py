import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from dotenv import load_dotenv

import threading, gc
from fastapi import Request
LLM_LOCK = threading.Lock()


def rebuild_llm():
    global _llm
    _llm = None
    gc.collect()
    ensure_llm_loaded()

# ========= Config via ENV (isi .env atau export) =========
load_dotenv()

MODEL_DIR   = Path(os.environ["MODEL_DIR"])
MODEL_FILE  = os.environ["MODEL_FILE"]
HF_REPO_ID  = os.environ["HF_REPO_ID"]
HF_FILENAME = os.environ["HF_FILENAME"]

N_THREADS   = int(os.environ["N_THREADS"])
N_CTX       = int(os.environ["N_CTX"])

# ========= App =========
app = FastAPI(title="FastAPI + llama.cpp (CPU)", version="0.1.0")

_llm: Optional[Llama] = None   # singleton LLM
_model_path: Optional[Path] = None  # resolved model path once prepared


# ========= Schemas =========
class ChatRequest(BaseModel):
    message: str = Field(..., description="Pertanyaan/prompt user")
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(256, ge=1, le=4096)
    system_prompt: Optional[str] = Field("You are a concise and helpful assistant.", description="System prompt opsional")

class ChatResponse(BaseModel):
    answer: str
    inference_seconds: float
    model_path: str


# ========= Helpers =========
def model_local_path() -> Path:
    return MODEL_DIR / MODEL_FILE

def model_is_ready() -> bool:
    return model_local_path().exists()

def prepare_model() -> Path:
    """
    Idempotent: jika file belum ada, unduh dari Hugging Face Hub.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = model_local_path()
    if target.exists():
        return target

    # Unduh dari HF Hub
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        token=os.environ["HF_TOKEN"],
    )

    # Jika nama file berbeda dari MODEL_FILE, samakan (copy/rename)
    dl_path = Path(downloaded)
    if dl_path.name != MODEL_FILE:
        # rename ke nama standar kita
        final_path = MODEL_DIR / MODEL_FILE
        if final_path.exists():
            return final_path
        dl_path.replace(final_path)
        return final_path

    return target

def ensure_llm_loaded() -> None:
    global _llm, _model_path
    if _llm is not None:
        return
    # pastikan model ada
    _model_path = prepare_model()
    # load sekali
    _llm = Llama(
        model_path=str(_model_path),
        n_threads=N_THREADS,
        n_batch=1024,    # <= tambah ini
        n_ctx=N_CTX,
        seed=1,
        logits_all=False,
        use_mmap=True,   # opsional: percepat load
        use_mlock=True   # opsional: hindari swap
    )


# ========= Endpoints =========
@app.get("/health")
def health():
    status = {
        "model_expected_path": str(model_local_path()),
        "model_ready": model_is_ready(),
        "llm_loaded": _llm is not None,
        "config": {
            "N_THREADS": N_THREADS,
            "N_CTX": N_CTX,
            "HF_REPO_ID": HF_REPO_ID,
            "HF_FILENAME": HF_FILENAME,
        }
    }
    return status

@app.post("/prepare_model")
def api_prepare_model():
    """
    Cek & siapkan model (download jika belum ada), tanpa memuat ke RAM.
    """
    path = prepare_model()
    return {"ok": True, "model_path": str(path), "exists": path.exists()}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Auto-prepare & auto-load model jika belum ada.
    """
    try:
        ensure_llm_loaded()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyiapkan/memuat model: {e}")

    assert _llm is not None
    assert _model_path is not None

    # format chat-completion ala llama_cpp
    messages = [
        {"role": "system", "content": req.system_prompt or ""},
        {"role": "user", "content": req.message},
    ]

    t0 = time.perf_counter()
    try:
        out = _llm.create_chat_completion(
            messages=messages,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            repeat_penalty=1.1,
        )
        answer = out["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    t1 = time.perf_counter()

    return ChatResponse(
        answer=answer,
        inference_seconds=round(t1 - t0, 3),
        model_path=str(_model_path)
    )

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(req: GenerateRequest, request: Request):
    global _llm
    try:
        ensure_llm_loaded()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyiapkan/memuat model: {e}")

    if _llm is None:
        raise HTTPException(status_code=400, detail="Model belum dimuat. Jalankan /prepare_model terlebih dahulu.")

    # === Kebijakan saat sibuk ===
    # - Kalau ingin request baru langsung ditolak saat masih ada proses: pakai acquire(blocking=False)
    # - Kalau ingin antri: pakai acquire() (blocking=True)
    if not LLM_LOCK.acquire(blocking=False):
        # model sedang generate untuk request lain → minta klien coba lagi
        raise HTTPException(status_code=409, detail="Model sedang memproses permintaan lain. Coba lagi.")

    start_time = time.time()
    try:
        # ===== Pengaturan generasi =====
        max_per_round = 512      # kecilkan agar responsif terhadap cancel
        max_rounds    = 3        # total kira-kira 1536 token
        stop_tokens   = None     # biarkan EOS internal

        # --- Batasi prompt berdasarkan token agar tidak overflow n_ctx ---
        toks = _llm.tokenize(req.prompt.encode("utf-8"))
        n_ctx = getattr(_llm, "n_ctx", lambda: 2048)()
        margin = max_per_round + 64
        if len(toks) > (n_ctx - margin):
            toks = toks[-(n_ctx - margin):]
            prompt_text = _llm.detokenize(toks).decode("utf-8", errors="ignore")
        else:
            prompt_text = req.prompt

        accumulated   = ""
        finish_reason = None

        for _ in range(max_rounds):
            # === panggilan sinkron per-batch; aman terhadap cancel di _antara_ putaran ===
            out = _llm.create_completion(
                prompt=prompt_text,
                max_tokens=max_per_round,
                temperature=0.3,
                top_p=0.9,
                stop=stop_tokens,
                repeat_penalty=1.05,
            )
            piece = out["choices"][0]["text"]
            accumulated += piece
            finish_reason = out["choices"][0].get("finish_reason")

            # Cek apakah klien sudah disconnect
            if await request.is_disconnected():
                # Untuk mencegah state korup dipakai lagi → reset model
                rebuild_llm()
                # 499: Client Closed Request (banyak server pakai kode ini)
                raise HTTPException(status_code=499, detail="Client menutup koneksi saat generate.")

            # Selesai normal (stop/EOS)
            if finish_reason and finish_reason != "length":
                break

            # Lanjut putaran berikutnya: tambahkan hasil ke konteks, tapi tetap jaga n_ctx
            next_prompt = prompt_text + piece
            toks2 = _llm.tokenize(next_prompt.encode("utf-8"))
            if len(toks2) > (n_ctx - margin):
                toks2 = toks2[-(n_ctx - margin):]
                prompt_text = _llm.detokenize(toks2).decode("utf-8", errors="ignore")
            else:
                prompt_text = next_prompt

        cleaned_answer = clean_output(accumulated)

    except HTTPException:
        # propagasikan 409/499/500 yang sudah kita set sendiri
        raise
    except Exception as e:
        # Jika ada error di tengah generate, reset agar sesi berikutnya bersih
        rebuild_llm()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        LLM_LOCK.release()

    elapsed_time = round(time.time() - start_time, 2)
    return {
        "output": cleaned_answer,
        "finish_reason": finish_reason or "unknown",
        "inference_time_seconds": elapsed_time
    }
    
import re

def clean_output(text: str) -> str:
    """
    Bersihkan output LLM tanpa memotong isinya:
    - Hapus blok kode bertanda ```...```
    - Hilangkan prefix seperti 'Answer:' di awal baris
    - Rapikan spasi berlebih per baris
    - Pertahankan pemisah paragraf (newlines)
    """
    # hapus fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    # hilangkan prefix umum (Answer:, Jawaban:, Response:) di awal baris
    text = re.sub(r"(?mi)^\s*(answer|jawaban|response)\s*:\s*", "", text)

    # rapikan setiap baris tapi pertahankan newline antar paragraf
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    # buang baris kosong beruntun → sisakan maksimal satu kosong
    cleaned_lines = []
    last_blank = False
    for ln in lines:
        if ln == "":
            if not last_blank:
                cleaned_lines.append("")
            last_blank = True
        else:
            cleaned_lines.append(ln)
            last_blank = False

    return "\n".join(cleaned_lines).strip()