"""Photos of ONE product -> brand + product name.

    from brand_extractor import extract_product
    result = extract_product(["front.jpg", "back.jpg"], category="grocery")
    result["brand"], result["product_name"]

Photos can be given in any order and with any file names: the rules never treat a photo as
special because of its position (the front is detected from what is printed on it).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from . import rules
from .ocr import PaddleOCRVL

CATEGORIES = ("grocery", "electronics")
_engine: PaddleOCRVL | None = None
_engine_lock = threading.Lock()


def get_engine() -> PaddleOCRVL:
    """One shared engine per process (the model is loaded once, on first use)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = PaddleOCRVL()
        return _engine


def extract_product(image_paths: list[str | Path], category: str = "grocery",
                    engine: PaddleOCRVL | None = None, debug: bool = False, on_photo=None,
                    names: list[str] | None = None) -> dict:
    """Run OCR on every photo, then the rules over all of them together.

    category: "grocery" (packaged goods) or "electronics"; it picks the product-name strategy.
    on_photo: optional callback(done, total) for progress reporting.
    names: display names for the photos (default: their file names).
    """
    category = category.lower()
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    if not image_paths:
        raise ValueError("no photos given")
    engine = engine or get_engine()
    t0 = time.time()
    spot_recs, ocr_recs, photos = [], [], []
    for i, path in enumerate(image_paths, 1):
        # Internal ids "Image_<n>_..." are what the rules use to tell photos apart; they carry
        # no meaning beyond identity (no rule favours any index).
        pid = f"Image_{i}_p"
        r = engine.read(path, name=f"{i}")
        spot_recs.append({"image": pid, "lines": r["lines"]})
        ocr_recs.append({"image": pid, "text": r["text"]})
        photos.append({"file": names[i - 1] if names else Path(path).name, "ocr_text": r["text"], "lines": len(r["lines"]),
                       "seconds": round(r["spot_s"] + r["ocr_s"], 2)})
        if on_photo:
            on_photo(i, len(image_paths))
    t_rules = time.time()
    out = rules.extract(spot_recs, ocr_recs, category.title(), debug=debug)
    result = {
        "brand": out["brand"],
        "product_name": out["product_name"],
        "category": category,
        "photos": photos,
        "seconds": round(time.time() - t0, 2),
        "rules_seconds": round(time.time() - t_rules, 3),
    }
    if debug:
        result["debug"] = out.get("debug")
    return result
