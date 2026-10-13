"""Brand + product name from product photos: PaddleOCR-VL-1.5 text reading + rules."""

from .pipeline import CATEGORIES, extract_product, get_engine

__all__ = ["CATEGORIES", "extract_product", "get_engine"]
