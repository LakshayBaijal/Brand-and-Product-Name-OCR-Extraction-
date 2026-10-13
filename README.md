# Brand & Product Name Extractor

Reads every photo of a product with **PaddleOCR-VL 1.5** and picks the **brand** and the
**product name** with hand-written rules. No LLM, no cloud calls: everything runs locally.

- Web app: add many products at once (each with its own photos), watch them process, export CSV / JSON
- Command line and Python API for scripts and batch jobs
- Photos can be in any order and have any file names

> Runs on **Apple Silicon Macs** (M1 or later): PaddleOCR-VL runs through Apple's MLX.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Open http://127.0.0.1:8000. The first start downloads the model (~0.9 GB) from Hugging Face;
after that it loads in a few seconds.

## Web app

1. **Add products.** Drop a folder with one subfolder per product, drop loose photos (grouped
   by name: `Image_1_X.jpg` / `Image_2_X.jpg` or `shampoo_1.jpg` / `shampoo_2.jpg` become one
   product each), or build products by hand with **+ Empty product**. Rename products, set the
   category, and add or remove photos before running.
2. **Extract.** Products are queued and processed one at a time; each row shows which photo it
   is reading.
3. **Results.** Brand, product name and time per product, the text read from each photo, and
   **Download CSV / JSON**.

Pick the right **category** per product: `grocery` (packaged goods: food, drinks, personal
care) or `electronics`. It decides how the product name is built.

## Command line

```bash
# one product, all its photos, any order  -> JSON
.venv/bin/python -m brand_extractor front.jpg back.jpg side.jpg --category grocery

# many products: a folder with one subfolder per product  -> results.csv
.venv/bin/python -m brand_extractor --folder products/ --category electronics --out results.csv
```

Paths with spaces need quotes.

## Python

```python
from brand_extractor import extract_product

r = extract_product(["front.jpg", "back.jpg"], category="grocery")
print(r["brand"], "|", r["product_name"])
# r["photos"] has the text read from each photo and its time; r["seconds"] the total
```

## How it works

```
photos ──► PaddleOCR-VL 1.5, two passes per photo ──► rules ──► brand + product name
             "Spotting:"  text lines + boxes + confidence
             "OCR:"       plain text in reading order
```

**OCR** (`brand_extractor/ocr.py`). PaddleOCR-VL is a 0.9B document-reading model that only
accepts task prompts. `Spotting:` returns each line with its position (that gives font size
and placement), `OCR:` spells better and keeps spaces, so both are used. Settings, measured
on an M5 with 16 GB: 6-bit weights (0.84 GB), 768 vision tokens per photo (1.8 GB peak memory),
photos capped at 1280 px, and a loop detector that stops runaway repetition ("GLUCOSE GLUCOSE
…") early.

**Brand** (`brand_extractor/rules.py`). Every line that could be a brand is scored. Strongest
signals: printed on many of the product's photos (the brand is on every side of a pack,
claims and flavours are not), large type, top of a front-looking photo, a ® / ™ mark, unusual
capitalisation (boAt, iPhone), and appearing right before a product line ("ASUS Vivobook").
Lines are rejected or penalised when they are regulatory text (ingredients, FSSAI, net
quantity, manufacturer addresses), nutrient names, generic words ("ENERGY DRINK", "WIRELESS"),
other companies' technology brands (Windows, Intel, Bluetooth), feature headings with a
sentence under them, or nutrition-table rows. Whether a photo is a "front" is judged from what
is printed on it, never from its position in the upload.

**Product name.** Grocery: the brand, then the other large lines of the front-looking photo,
plus flavour / variant lines. Electronics: the brand plus the model that recurs across the
photos ("WH-CH520", "Tune 770NC", "X6 Pro").

## Accuracy and speed

Measured on an internal set of 100 products (50 grocery, 50 electronics, 429 photos):

| | Brand correct | Brand exactly right |
|---|---|---|
| All | 74% | 63% |
| Grocery | 90% | 72% |
| Electronics | 58% | 53% |

"Correct" allows small spelling differences ("O'cean" vs "o'cean one8"); "exactly right"
means identical after ignoring case and punctuation.

- The rules were developed while looking at these same products, so expect lower numbers on
  new products.
- Typical misses: brands that appear only as a logo (OCR returns no text), brands made of
  ordinary words ("Made From Plants"), and laptop listing images where the brand is a small
  footer.
- Speed on an M5: about 2.5-4 s per photo, so 10-20 s for a typical 4-photo product;
  products are processed one at a time.

## Project layout

```
app.py                  web server (FastAPI) + job queue
web/index.html          the web page (no external libraries)
brand_extractor/
  ocr.py                PaddleOCR-VL engine (MLX), one dedicated model thread
  rules.py              brand / product-name rules
  pipeline.py           extract_product(): photos -> result
  __main__.py           command line
```

## Notes

- The model is `mlx-community/PaddleOCR-VL-1.5-6bit`, an MLX conversion of
  [PaddlePaddle/PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle) (Apache-2.0).
- The rules use the macOS word list at `/usr/share/dict/words` to tell plain English words
  from brand names.
- Uploaded photos are kept only until their product is processed; results stay in memory
  until you clear them or stop the app.
