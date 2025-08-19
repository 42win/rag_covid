import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from dotenv import load_dotenv

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
        n_ctx=N_CTX,
        seed=1,
        logits_all=False
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
def generate_text(req: GenerateRequest):
    global _llm
    try:
        ensure_llm_loaded()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyiapkan/memuat model: {e}")

    if _llm is None:
        raise HTTPException(status_code=400, detail="Model belum dimuat. Jalankan /prepare_model terlebih dahulu.")

    start_time = time.time()
    try:
        # gunakan create_completion tanpa n_batch
        output = _llm.create_completion(
            prompt=req.prompt,
            max_tokens=250,
            stop=["</s>", "ANSWER:"],
            temperature=0.3,
            top_p=0.9
        )

        # ambil dan bersihkan jawaban
        raw_answer = output["choices"][0]["text"].strip()
        cleaned_answer = clean_output(raw_answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_time = round(time.time() - start_time, 2)
    return { 
        "output": cleaned_answer,
        "inference_time_seconds": elapsed_time
    }


def clean_output(text: str) -> str:
    """
    Membersihkan output LLM:
    - Hapus blok kode atau tanda ```
    - Ambil hanya paragraf pertama
    - Hilangkan spasi ganda
    """
    import re
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # hapus blok kode
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)  # normalisasi spasi
    return text.split(".")[0].strip() + "."


from build_kb import run_build_kb

@app.post("/build-kb")
def api_build_kb():
    return run_build_kb()  # atau kirim argumen sesuai kebutuhan 

# ===== Startup: inisialisasi index & model =====

import searching as S   

def _db_path():
    return Path(os.environ.get("CHROMA_PATH",
           Path(__file__).resolve().parent.parent / "rag_covid" / "chroma_db_e5"))

@app.on_event("startup")
def _startup():
    try:
        S.init_search(chroma_path=_db_path(), collection_name=os.environ.get("COLLECTION_NAME","covid_docs_e5"))
        print("[startup] ready:", S.is_ready())
    except Exception as e:
        print("Init error:", e)

class QueryBody(BaseModel):
    query: str = Field(..., description="Pertanyaan pengguna")
    top_k_factoid: int = 1
    return_dict: bool = True

def _ensure_ready():
    if not S.is_ready():
        S.init_search(chroma_path=_db_path(), collection_name=os.environ.get("COLLECTION_NAME","covid_docs_e5"))
    if not S.is_ready():
        raise RuntimeError("Belum init_search(). Pastikan DB sudah dibangun dan path benar.")

@app.post("/search")
def api_search(body: QueryBody):
    try:
        _ensure_ready()  # lazy init saat request
        result = S.answer_query_auto(
            body.query,
            top_k_factoid=body.top_k_factoid,
            return_dict=body.return_dict,
        )
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")


from pydantic import BaseModel, Field
import generate_llm as G

class RagBody(BaseModel):
    query: str = Field(..., description="Pertanyaan pengguna")
    top_k_factoid: int = 1
    # llm_url: Optional[str] = None

@app.post("/rag-answer")
def rag_answer(body: RagBody):
    try:
        _ensure_ready()  # pastikan searching siap
        result = G.run_rag_answer(
            query=body.query,
            top_k_factoid=body.top_k_factoid,
            # llm_url=body.llm_url,  # boleh None -> pakai ENV/default
        )
        return {"ok": True, "result": result}
    except Exception as e:
        # konsisten dengan error style sebelumnya
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")