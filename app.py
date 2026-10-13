"""Web app: add several products (each with its own photos), get brand + product name for each.

    python app.py            # then open http://127.0.0.1:8000
    python app.py --port 9000 --host 0.0.0.0

Products are processed one at a time in the order they were submitted (one model, one
queue). Photos are deleted once their product is processed; only small thumbnails and the
results stay in memory until the app stops or you clear them.
"""

from __future__ import annotations

import argparse
import io
import queue
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps

from brand_extractor import CATEGORIES, extract_product, get_engine

HERE = Path(__file__).resolve().parent
UPLOADS = Path(tempfile.mkdtemp(prefix="brand_extractor_uploads_"))
MAX_PHOTOS = 20
MAX_PHOTO_MB = 25

app = FastAPI(title="Brand & Product Name Extractor")
jobs: dict[str, dict] = {}
order: list[str] = []
work: "queue.Queue[str]" = queue.Queue()
model_state = {"status": "loading", "seconds": None, "error": None}


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("paths", "thumbs")} | {"n_photos": len(job["thumbs"])}


def _load_model() -> None:
    try:
        model_state["seconds"] = round(get_engine().warm_up(), 1)
        model_state["status"] = "ready"
    except Exception as e:  # e.g. no network on first run
        model_state.update(status="error", error=f"{type(e).__name__}: {e}")


def _worker() -> None:
    while True:
        jid = work.get()
        job = jobs.get(jid)
        if job is None or job["status"] != "queued":
            continue
        job.update(status="running", started=time.time())
        try:
            res = extract_product(job["paths"], job["category"], names=job["photo_names"],
                                  on_photo=lambda done, total: job.update(done_photos=done))
            job.update(status="done", result=res, seconds=res["seconds"])
        except Exception as e:
            job.update(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            shutil.rmtree(UPLOADS / jid / "photos", ignore_errors=True)  # keep thumbnails only
            job["paths"] = []


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_model, daemon=True).start()
    threading.Thread(target=_worker, daemon=True).start()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "web" / "index.html")


@app.get("/api/status")
def status() -> dict:
    counts = {s: sum(j["status"] == s for j in jobs.values()) for s in ("queued", "running", "done", "error")}
    return {"model": model_state, "categories": list(CATEGORIES), "counts": counts,
            "limits": {"max_photos": MAX_PHOTOS, "max_photo_mb": MAX_PHOTO_MB}}


@app.post("/api/products")
async def add_product(name: str = Form(...), category: str = Form("grocery"),
                      photos: list[UploadFile] = File(...)) -> dict:
    category = category.lower()
    if category not in CATEGORIES:
        raise HTTPException(400, f"category must be one of {list(CATEGORIES)}")
    if not 1 <= len(photos) <= MAX_PHOTOS:
        raise HTTPException(400, f"give 1-{MAX_PHOTOS} photos per product")
    jid = uuid.uuid4().hex[:10]
    pdir, tdir = UPLOADS / jid / "photos", UPLOADS / jid / "thumbs"
    pdir.mkdir(parents=True)
    tdir.mkdir()
    paths, thumbs = [], []
    for i, up in enumerate(photos, 1):
        data = await up.read()
        if len(data) > MAX_PHOTO_MB * 1024 * 1024:
            shutil.rmtree(UPLOADS / jid, ignore_errors=True)
            raise HTTPException(400, f"{up.filename}: larger than {MAX_PHOTO_MB} MB")
        try:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
            im.load()
        except Exception:
            shutil.rmtree(UPLOADS / jid, ignore_errors=True)
            raise HTTPException(400, f"{up.filename}: not an image this app can read")
        suffix = Path(up.filename or "").suffix.lower() or ".jpg"
        p = pdir / f"{i}{suffix}"
        p.write_bytes(data)
        th = im.convert("RGBA").convert("RGB") if im.mode in ("P", "LA", "RGBA") else im.convert("RGB")
        th.thumbnail((240, 240))
        t = tdir / f"{i}.jpg"
        th.save(t, "JPEG", quality=80)
        paths.append(str(p))
        thumbs.append({"file": up.filename, "path": str(t)})
    job = {"id": jid, "name": name.strip() or f"Product {len(order) + 1}", "category": category,
           "status": "queued", "done_photos": 0, "photo_names": [t["file"] for t in thumbs],
           "created": time.time(), "result": None, "error": None, "seconds": None,
           "paths": paths, "thumbs": thumbs}
    jobs[jid] = job
    order.append(jid)
    work.put(jid)
    return _public(job)


@app.get("/api/products")
def list_products() -> list[dict]:
    return [_public(jobs[j]) for j in order if j in jobs]


@app.get("/api/products/{jid}/thumb/{n}")
def thumb(jid: str, n: int) -> Response:
    job = jobs.get(jid)
    if not job or not 1 <= n <= len(job["thumbs"]):
        raise HTTPException(404)
    return FileResponse(job["thumbs"][n - 1]["path"], media_type="image/jpeg")


@app.delete("/api/products/{jid}")
def delete_product(jid: str) -> dict:
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404)
    if job["status"] == "running":
        raise HTTPException(409, "this product is being processed; wait for it to finish")
    jobs.pop(jid)
    order.remove(jid)
    shutil.rmtree(UPLOADS / jid, ignore_errors=True)
    return {"deleted": jid}


@app.delete("/api/products")
def clear_finished() -> dict:
    gone = [j for j in order if jobs[j]["status"] in ("done", "error")]
    for j in gone:
        delete_product(j)
    return {"deleted": len(gone)}


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"Open http://{a.host}:{a.port}  (the model loads in the background on first start)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
