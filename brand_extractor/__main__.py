"""Command line.

  One product (all its photos):
    python -m brand_extractor front.jpg back.jpg side.jpg --category grocery

  Many products - a folder with one subfolder per product:
    python -m brand_extractor --folder products/ --category electronics --out results.csv

Prints JSON for one product; writes CSV (and prints a line per product) for a folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .pipeline import CATEGORIES, extract_product, get_engine

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def photos_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m brand_extractor",
                                 description="Brand + product name from product photos (PaddleOCR-VL + rules).")
    ap.add_argument("photos", nargs="*", type=Path, help="photos of ONE product, any order")
    ap.add_argument("--folder", type=Path, help="folder with one subfolder of photos per product")
    ap.add_argument("--category", choices=CATEGORIES, default="grocery")
    ap.add_argument("--out", type=Path, default=Path("results.csv"), help="CSV output for --folder")
    ap.add_argument("--debug", action="store_true", help="include the scored brand candidates")
    args = ap.parse_args()

    if bool(args.photos) == bool(args.folder):
        ap.error("give photos of one product, or --folder (not both)")
    missing = [str(p) for p in args.photos if not p.is_file()]
    if missing:
        ap.error("photo(s) not found: " + ", ".join(missing) +
                 "\n(paths with spaces must be in quotes)")

    print(f"loading model ... {get_engine().warm_up():.1f} s", file=sys.stderr)

    if args.photos:
        r = extract_product(args.photos, args.category, debug=args.debug)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return

    products = [d for d in sorted(args.folder.iterdir()) if d.is_dir() and photos_in(d)]
    if not products:
        ap.error(f"no subfolders with photos in {args.folder}")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product", "category", "brand", "product_name", "photos", "seconds"])
        for n, d in enumerate(products, 1):
            r = extract_product(photos_in(d), args.category)
            w.writerow([d.name, args.category, r["brand"], r["product_name"], len(r["photos"]), r["seconds"]])
            f.flush()
            print(f"[{n}/{len(products)}] {d.name}: {r['brand']!r} / {r['product_name']!r} ({r['seconds']} s)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
