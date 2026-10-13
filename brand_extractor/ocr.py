"""PaddleOCR-VL-1.5 reader (MLX, Apple Silicon): two passes per photo.

  "Spotting:"  text lines with a box (8 x <|LOC_n|> tokens, 0-1000 scale) and a per-line
               confidence from the token log-probs -> font size and position for the rules
  "OCR:"       plain text in reading order -> better spelling, keeps spaces

Settings are the ones measured on an M5 / 16 GB (see README):
  6-bit weights (0.84 GB), 768 vision tokens per photo (peak memory 1.8 GB), photos capped
  at 1280 px, and a loop detector that stops "GLUCOSE GLUCOSE ..." runaways early.

Threading: MLX's GPU stream belongs to the thread that first imports mlx_vlm, and a web
server handles requests on many threads. Every model call (import, load, generate) runs on
ONE dedicated worker thread owned by this module.
"""

from __future__ import annotations

import math
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

MODELS = {"6bit": "mlx-community/PaddleOCR-VL-1.5-6bit", "bf16": "mlx-community/PaddleOCR-VL-1.5-bf16"}
LINE_RE = re.compile(r"^(.*?)((?:<\|LOC_\d+\|>){8})\s*$")
LOC_RE = re.compile(r"<\|LOC_(\d+)\|>")

_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def looping(ids: list[int], window: int = 48, max_period: int = 12) -> bool:
    """True when the last `window` tokens repeat a short period: the model is stuck in a
    loop ("1kg 1kg 1kg", "GLUCOSE GLUCOSE") and nothing new will follow."""
    if len(ids) < window + 16:
        return False
    tail = ids[-window:]
    for p in range(1, max_period + 1):
        reps = window // p
        if tail[-reps * p:] == tail[-p:] * reps:
            return True
    return False


def dedupe_runaway(text: str, max_repeats: int = 3) -> str:
    """Keep at most `max_repeats` copies of any identical line."""
    seen: dict[str, int] = {}
    out = []
    for line in text.splitlines():
        k = line.strip().lower()
        seen[k] = seen.get(k, 0) + 1
        if seen[k] <= max_repeats:
            out.append(line)
    return "\n".join(out)


def parse_spotting(tok, ids: list[int], lps: list[float]) -> list[dict]:
    """Spotting output -> [{text, box [x0,y0,x1,y1], quad, height, width, conf}]."""
    groups: list[list[float]] = [[]]
    for i, lp in zip(ids, lps):
        piece = tok.decode([i])
        if "\n" in piece:
            groups.append([])
        elif not piece.startswith("<|LOC_") and piece.strip():
            groups[-1].append(lp)
    confs = [math.exp(sum(g) / len(g)) if g else None for g in groups]
    text_lines = [l for l in tok.decode(ids, skip_special_tokens=False).replace("</s>", "").split("\n") if l.strip()]
    if len(confs) != len(text_lines):  # alignment lost -> no confidence rather than a wrong one
        confs = [None] * len(text_lines)
    lines = []
    for raw, conf in zip(text_lines, confs):
        m = LINE_RE.match(raw)
        if not m:
            lines.append({"text": raw.strip(), "box": None, "quad": None, "height": None, "width": None, "conf": conf})
            continue
        q = [int(v) for v in LOC_RE.findall(m.group(2))]
        xs, ys = q[0::2], q[1::2]
        box = [min(xs), min(ys), max(xs), max(ys)]
        lines.append({"text": m.group(1).strip(), "box": box, "quad": q,
                      "height": box[3] - box[1], "width": box[2] - box[0], "conf": conf})
    return lines


class PaddleOCRVL:
    """Lazy-loading PaddleOCR-VL-1.5. Thread-safe: calls are serialised onto the MLX thread."""

    def __init__(self, model: str = "6bit", vision_tokens: int = 768, max_side: int = 1280,
                 spot_tokens: int = 1200, ocr_tokens: int = 1024):
        self.repo = MODELS.get(model, model)
        self.vision_tokens, self.max_side = vision_tokens, max_side
        self.spot_tokens, self.ocr_tokens = spot_tokens, ocr_tokens
        self._m = None
        self._tmp = Path(tempfile.mkdtemp(prefix="brand_extractor_"))

    # ---------------------------------------------------------------- MLX thread only
    def _load(self):
        if self._m is None:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            model, processor = load(self.repo, trust_remote_code=True)
            processor.image_processor.max_pixels = self.vision_tokens * 28 * 28
            self._m = (model, processor, load_config(self.repo))
        return self._m

    def _generate(self, prompt: str, image: str, max_tokens: int):
        import mlx.core as mx
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor, config = self._load()
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        ids, lps = [], []
        for r in stream_generate(model, processor, formatted, image=[image],
                                 max_tokens=max_tokens, temperature=0.0):
            ids.append(r.token)
            lps.append(float(r.logprobs.reshape(-1)[r.token]))
            if len(ids) % 16 == 0 and looping(ids):
                break
        mx.clear_cache()
        return ids, lps

    def _read(self, image_path: str) -> dict:
        _, processor, _ = self._load()
        tok = processor.tokenizer
        t0 = time.time()
        ids, lps = self._generate("Spotting:", image_path, self.spot_tokens)
        t1 = time.time()
        lines = parse_spotting(tok, ids, lps)
        ids, _ = self._generate("OCR:", image_path, self.ocr_tokens)
        text = dedupe_runaway(tok.decode(ids, skip_special_tokens=True).strip())
        return {"lines": lines, "text": text,
                "spot_s": round(t1 - t0, 2), "ocr_s": round(time.time() - t1, 2)}

    # ---------------------------------------------------------------- any thread
    def warm_up(self) -> float:
        """Download (first run) and load the model; returns seconds taken."""
        t = time.time()
        _worker.submit(self._load).result()
        return time.time() - t

    def prepare(self, src: str | Path, name: str) -> str:
        """EXIF-upright RGB JPEG capped at max_side px (phone photos are often 4000 px)."""
        im = ImageOps.exif_transpose(Image.open(src))
        if im.mode in ("P", "LA", "RGBA"):
            im = im.convert("RGBA")
        im = im.convert("RGB")
        im.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
        dst = self._tmp / f"{name}.jpg"
        im.save(dst, "JPEG", quality=92)
        return str(dst)

    def read(self, image_path: str | Path, name: str = "photo") -> dict:
        """Both passes on one photo -> {lines, text, spot_s, ocr_s}."""
        prepared = self.prepare(image_path, name)
        try:
            return _worker.submit(self._read, prepared).result()
        finally:
            Path(prepared).unlink(missing_ok=True)
