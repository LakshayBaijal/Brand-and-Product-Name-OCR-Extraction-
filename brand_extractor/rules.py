"""Stage 2, rule-based: brand + product name from PaddleOCR-VL-1.5 output only. No LLM,
no second model - the production-on-small-CPU path.

Inputs (both from PaddleOCR-VL-1.5, see run_spotting.py / run_paddleocr_vl.py):
  spotting.jsonl  text lines WITH boxes (font size, position) and token confidence
  ocr.jsonl       plain "OCR:" text: second spelling vote, recurrence counting, and a
                  reading-order fallback (Spotting sometimes drops spaces / boxes)

BRAND - signals, strongest first (measured on the eval set, not assumed):
  1. RECURRENCE  the brand is printed on every panel/photo; variant names, claims and
                 flavours are not. Counted inside ANY line of ANY image, fuzzily, so
                 "Horlcks" and "Advantages of Horlicks Diabetes Plus" both count.
  2. SIZE        font size = min(box width, box height) relative to the largest text in
                 that image (a vertical logo's font size is its WIDTH).
  3. POSITION    top of the front panel; the topmost large line gets a bonus.
  4. SHAPE       1-3 words; the first word of a listing title ("JBL HEADPHONES APP",
                 "ASUS Vivobook") is a candidate by e-commerce convention; phrases made
                 only of plain dictionary words ("Pause Play", "Hours") are penalised.
  CONFIDENCE is NOT a brand signal: stylised logos ("O.cean" 0.29) score far lower than
  body text (~1.0). It only breaks ties between spellings of the same line.
  Hard filters: stop-list phrases (ingredients, net quantity, FSSAI, manufactured by...),
  mostly-numeric lines, sentences, spec tokens ("85dBA"), and lines made only of generic
  words (ENERGY DRINK, SUGAR FREE, NEW, WIRELESS, ANC...). Generic words are stripped off
  the ends of mixed lines: "NEW STING ENERGY" -> "Sting", "GATORADE SPORTS DRINK" -> "Gatorade".

PRODUCT NAME - category aware (the category is known from the catalog):
  Grocery      brand + the other large non-claim lines of the front panel in reading
               order + flavour/variant lines from any panel. Fallback: "OCR:" reading order.
  Electronics  brand + the MODEL line: a short line holding a model token (letters+digits
               such as "770NC", "WH-CH520", "Nord 3", "Airdopes 500") that recurs across
               images and is not a spec ("256GB", "5000mAh", "13th Gen").

Entry point: extract(spot_recs, ocr_recs, category) -> {"brand", "product_name"}.
Called by brand_extractor.pipeline; never reads or writes files.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher


# =============================================================================
# Text utilities
# =============================================================================


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def words(s: str) -> list[str]:
    return [w for w in re.split(r"[\s/|]+", (s or "").strip()) if w]


def clean(s: str) -> str:
    return (s or "").strip(" .,:;-*'\"®™•·")


def alpha_ratio(s: str) -> float:
    letters = sum(c.isalpha() for c in s)
    return letters / max(1, sum(not c.isspace() for c in s))


def similar(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.8


def pretty(s: str) -> str:
    """ALL-CAPS tokens to Title case; leave mixed case (Glucon-D, boAt, iPhone) alone."""
    return " ".join(w.title() if w.isupper() and len(w) > 1 and not re.search(r"\d", w) else w
                    for w in words(s))


# =============================================================================
# Lexicons
# =============================================================================

# Phrases that can be printed large on a pack / listing but are never a brand or name part.
STOP_PHRASES = [
    "ingredient", "nutrition", "nutritional", "information", "net quantity", "net qty", "net wt",
    "net weight", "mrp", "m.r.p", "fssai", "lic. no", "lic no", "licence", "license",
    "manufactured", "manufacturer", "marketed", "packed", "imported", "importer", "brand owner",
    "address", "customer care", "consumer care", "contact", "email", "e-mail", "www.", ".com",
    "toll free", "batch", "mfg", "mfd", "exp", "best before", "use by", "use before", "date",
    "store in", "keep", "refrigerate", "shake well", "serve", "serving", "per 100", "per serve",
    "kcal", "rda", "energy (", "protein", "carbohydrate", "sugars", "total fat", "sodium",
    "vitamin", "calcium", "caffeine", "trademark", "registered", "all rights", "copyright",
    "not recommended", "recommended for", "pregnant", "lactating", "children", "allergen",
    "contains", "approx", "guideline", "offence", "offense", "punishable", "proprietary food",
    "beverage mix", "for special dietary", "helps", "manage", "support", "reduce", "clinically",
    "proven", "added sugar", "zero added", "no added", "hydration", "vitalizes", "wholesale",
    "price", "inclusive", "taxes", "incl.", "scan", "qr", "barcode", "recycle", "dispose",
    "warning", "caution", "disclaimer", "limitation", "as per", "data", "share", "survey",
    "in the world", "india's no", "no.1", "no. 1", "trusted", "since 19", "since 20",
    "based on", "tested", "testing", "reminder", "conditions apply", "visit the store",
    "bought in", "past month", "★", "rating", "review", "sponsored", "free delivery",
    "in stock", "add to cart", "buy now", "emi", "offer", "discount", "% off", "coupon",
    "with ", "without ", "made in", "caffeinated", "minerals", "nutrients", "mcg", "storage",
    "instructions", "lot ", "lot:", "shelf life", "expiry", "product of", "country of origin",
    "made with", "chilled", "warranty", "replacement", "return", "delivery", "seller",
    "quantity", "specification", "specs", "feature", "features", "description", "about this",
    "what's in the box", "in the box", "box contents", "user manual", "manual", "cable",
    "charger", "adapter", "processor", "graphics", "display", "screen", "resolution",
    "refresh rate", "storage", "memory", "operating system", "keyboard", "fingerprint",
    "weight", "dimension", "thickness", "thin and light", "sim", "dual sim", "camera",
    "rear", "front", "selfie", "megapixel", "battery", "mah", "fast charging", "charging",
    "playtime", "latency", "driver", "drivers", "frequency", "impedance", "sensitivity",
    "bluetooth 5", "bluetooth version", "ip rating", "water resistant", "sweat",
    "noise reduction", "noise cancel", "hear ", "listen", "enjoy", "immersive", "comfort",
    "lightweight", "all-day", "all day", "ultra", "zone out", "focus", "surroundings",
    "zero distractions", "sharpen", "music never", "even when", "pure sound", "sound of",
    "experience", "technology", "block", "giving you", "silence", "personal space",
    "advanced", "amplify", "voice", "journey", "painless", "breathable", "cushion",
    "distortion", "deepest", "treble", "detailed", "produce", "amazing", "powerful",
    "exceptional", "premium sound", "soft memory", "hands-free", "cvc", "hd calls",
    "upgraded", "feedback", "feedforward", "mode 0", "lights on", "swivel", "earcups",
    "easy operation", "plug in", "aux plug", "hi-fi", "hifi", "effects", "pop", "rock",
    "jazz", "vocal", "classic", "customisation", "customization", "simple buttons",
    "swift pair", "set reminder", "mute the world", "red dot", "reddot", "award",
    # company / legal-entity lines (manufacturer, marketer, licensee)
    " ltd", "limited", " pvt", " inc", " llc", "gmbh", "corporation", "corp.", "industries",
    "enterprises", "foods pvt", "beverages pvt", "wellness products", "consumer products",
    "healthcare", "pharma", "laboratories", "unilever", "nestle", "nestlé", "pepsico", "danone",
    "abbott", "zydus", "hindustan", "stokely", "mondelez", "itc limited", "dabur india",
    # nutrient names
    "taurine", "niacin", "riboflavin", "thiamine", "thiamin", "inositol", "glucuronolactone",
    "biotin", "folate", "folic", "pantothen", "pyridoxine", "cobalamin", "ascorbic", "retinol",
    "magnesium", "potassium", "phosphorus", "selenium", "chromium", "iodine", "copper",
    "manganese", "lutein", "choline", "carnitine", "creatine", "arginine", "leucine",
]

# Words that are fine inside a product name but on their own are never a brand.
GENERIC_WORDS = {
    "the", "a", "an", "of", "and", "&", "for", "in", "new", "now", "original", "classic",
    "regular", "premium", "pure", "fresh", "natural", "real", "100", "100%", "free", "zero",
    "no", "low", "high", "extra", "plus", "lite", "light", "edition", "limited", "special",
    "summer", "winter", "pack", "family", "value", "super", "saver", "combo", "drink", "drinks", "beverage",
    "energy", "sports", "sport", "instant", "health", "fuel", "power", "hydration",
    "electrolytes", "electrolyte", "fluids", "sugar", "sugarfree", "sugar-free", "caffeine",
    "flavour", "flavor", "flavoured", "flavored", "taste", "mix", "powder", "juice", "water",
    "milk", "tea", "coffee", "chocolate", "vanilla", "strawberry", "orange", "mango",
    "lemon", "lime", "berry", "mixed", "fruit", "fruits", "tangy", "ml", "l", "g", "gm", "kg",
    "mg", "can", "bottle", "jar", "pouch", "net", "wt", "qty", "size", "large", "small", "mini",
    "max", "pro", "diet", "vegetarian", "veg", "ready", "to", "eat", "cook", "is", "it", "your",
    "one", "nos", "pcs", "piece", "pieces", "x", "ones", "added", "contains", "natural",
    "oat", "almond", "soy", "protein", "nutrition", "nutritional", "supplement", "shake",
    "malt", "based", "food", "bar", "bars", "snack", "daily", "choco", "nut", "cream", "creamy",
    "rich", "kesar", "badam", "hazelnut", "banana", "caramel", "plant", "plants", "made",
    "from", "herbal", "ayurvedic", "immunity", "sharbat", "syrup", "aloe", "vera", "giloy",
    "amla", "karela", "soft", "cola", "soda", "creme", "caramel", "yummy", "tasty", "delicious",
    "herbs", "herb", "facts", "cubes",
    # electronics feature jargon printed large on listing infographics
    "anc", "noise", "cancelling", "cancellation", "canceling", "wireless", "wired", "mode",
    "aux", "mic", "mics", "microphone", "sound", "audio", "bass", "stereo", "hours", "hrs",
    "hour", "playtime", "battery", "charging", "charge", "port", "micro", "sd", "usb",
    "type-c", "hybrid", "active", "hi-res", "ldac", "pause", "play", "call", "calls", "key",
    "volume", "safe", "kids", "kid", "toddler", "dba", "db", "up", "compatible", "standby",
    "days", "fast", "pair", "pairing", "button", "buttons", "hear", "details", "listen",
    "longer", "immersive", "enjoy", "ambient", "aware", "adaptive", "true", "truly",
    "foldable", "over-ear", "on-ear", "in-ear", "headphones", "headphone", "earbuds",
    "earphones", "earphone", "neckband", "tws", "app", "smart", "control", "touch", "led",
    "lights", "light", "hearing", "protect", "bpa", "cool", "off", "on", "mp3", "mins", "min",
    "breathing", "connect", "connection", "connections", "multipoint", "upgraded", "feedback",
    "feedforward", "crystal", "clear", "hd", "fit", "comfort", "laptop", "notebook", "type",
    "signature", "resistance", "controls", "interface", "splash", "rgb", "leds", "playback",
    "ois", "dsp", "ipx4", "ipx5", "ipx7", "magnetic", "unboxing", "tm", "design", "ratio",
    "software", "hotkeys", "performance", "military", "grade", "nits", "level", "color",
    "name", "built-in", "kickstand", "aspect", "passed", "gpu", "cpu", "lcd", "turbo", "modes",
    "press", "single", "double", "triple", "shield", "ceramic", "aura", "blazing", "beast",
    "srgb", "dci-p3", "ntsc", "adobe", "nanoedge", "talkthru", "smooth", "seamless",
    "series", "gen", "ssd", "ram", "rom", "gb", "tb", "inch", "fhd", "oled", "ips", "included",
    "home", "student", "version", "gaming", "creator", "thin", "touchscreen", "5g", "4g",
    "lte", "dual", "quad", "octa", "phone", "smartphone", "mobile", "storage", "refurbished",
    "renewed", "unlocked",
}
# Listing colour names: a stop for the model phrase only ("Nord 3 5G (Misty Green..."),
# never stripped from a brand ("Red Bull").
COLOUR_WORDS = {"green", "blue", "red", "silver", "gold", "grey", "gray", "pink", "purple",
                "lavender", "titanium", "misty", "frozen", "vintage", "desert", "spectre",
                "supernova", "mercurial", "gem", "black", "white", "graphite", "midnight",
                "starlight", "cream", "beige", "yellow", "mint", "teal", "orange"}

# Third-party technology brands printed on electronics packaging, never the product brand.
CO_BRANDS = {
    "windows", "microsoft", "office", "mcafee", "norton", "intel", "core", "i3", "i5", "i7",
    "i9", "celeron", "pentium", "amd", "ryzen", "radeon", "nvidia", "geforce", "rtx", "gtx",
    "bluetooth", "dolby", "atmos", "alexa", "google", "siri", "android", "ios", "qualcomm",
    "snapdragon", "mediatek", "dimensity", "helio", "exynos", "aptx", "usb-c", "thunderbolt",
    "hdmi", "wifi", "wi-fi", "nfc", "amazon", "flipkart", "reddotaward", "evo", "iris", "xe",
    "jio", "airtel", "netflix", "spotify", "youtube", "whatsapp", "instagram", "facebook",
    "gorilla", "corning", "ultrawide", "amoled", "ddr4", "ddr5", "lpddr4", "lpddr5", "lpddr4x",
    "lpddr5x", "nvme", "pcie", "m.2", "sata", "uhd", "qhd", "wqhd", "wuxga", "sgb", "emmc",
}
GENERIC_WORDS |= CO_BRANDS
BRAND_GENERIC = {norm(w) for w in GENERIC_WORDS} - {"ones"}  # normalised: "hi-res" -> "hires"
GENERIC_LONG = {g for g in BRAND_GENERIC if len(g) >= 5}

try:  # a candidate made only of plain English words is unlikely to be a brand
    DICT_WORDS = {w.strip().lower() for w in open("/usr/share/dict/words") if len(w.strip()) >= 3}
except OSError:
    DICT_WORDS = set()

CONNECTORS = {"by", "and", "with", "for", "of", "or", "to", "in", "on", "your", "the", "is", "are", "up"}
UNIT_WORDS = {"ml", "l", "g", "gm", "gms", "kg", "mg", "mcg", "litre", "liter", "ltr", "oz", "lb",
              "net", "wt", "qty", "pcs", "nos", "n", "x", "mm", "cm", "inch", "hz", "w", "v", "mah"}
VARIANT_RE = re.compile(r"\b(flavou?r|flavou?red)\b", re.I)
FLAVOUR_WORDS = {"chocolate", "vanilla", "strawberry", "orange", "mango", "lemon", "lime", "banana",
                 "caramel", "hazelnut", "kesar", "badam", "pista", "elaichi", "cardamom", "coffee",
                 "mocha", "butterscotch", "almond", "cookie", "cookies", "cream", "berry", "mixed",
                 "fruit", "apple", "grape", "guava", "litchi", "lychee", "peach", "pineapple", "rose",
                 "thandai", "masala", "jeera", "mint", "tangy", "classic", "original", "regular",
                 "plain", "unflavoured", "unflavored", "natural", "diet", "zero", "lite", "light",
                 "sugarfree", "sugar-free", "max", "plus", "pro", "gold", "royale", "royal", "creme"}


def is_flavour_line(t: str) -> bool:
    ws = [norm(w) for w in words(t)]
    return bool(ws) and len(ws) <= 3 and all(w in FLAVOUR_WORDS or w in {"flavour", "flavor", "flavoured", "flavored"} for w in ws)
VARIANT_SIZE = re.compile(r"\b\d+(\.\d+)?\s*(ml|l|g|gm|kg|mg)\b", re.I)

# Electronics: model tokens vs spec tokens.
# A model token has letters AND digits ("770NC", "WH-CH520", "Y28s", "E9", "M16"); a bare
# number ("15", "3") counts only right after a capitalised line name ("Vivobook 15", "Nord 3").
MODEL_TOKEN = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9][A-Za-z0-9\-+]{0,11}$")
BARE_NUMBER = re.compile(r"^\d{1,4}[A-Za-z]?$")
SPEC_TOKEN = re.compile(r"^\d+(\.\d+)?\s*(gb|tb|mb|mah|hz|khz|mhz|ghz|w|v|mm|cm|inch|in|hrs?|h|db|dba|mp|k|ms|nits|wh|whr|%|th|st|nd|rd|x|g|m|mins?|sec|s|fps|bit|ch|khz|kg|lbs?|oz)$", re.I)
MONTHS = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
          "january", "february", "march", "april", "june", "july", "august", "september",
          "october", "november", "december"}
UNIT_LIKE = {"mins", "min", "minutes", "hours", "hrs", "hour", "days", "mhz", "ghz", "bought",
             "times", "x", "sec", "seconds", "percent", "%", "mm", "cm", "inch", "inches",
             "gb", "tb", "mah", "hz", "w", "v", "db", "mp", "year", "years", "month", "months",
             "playing", "charge", "charging", "playtime", "standby", "ms"}


# =============================================================================
# Line-level predicates
# =============================================================================


STOP_TOKENS = {"fssai", "ingredients", "nutritional", "nutrition", "manufactured", "marketed",
               "trademark", "registered", "disclaimer", "flavouring", "substances", "identical",
               "maltodextrin", "emulsifier", "stabilizer", "stabiliser", "preservative", "antioxidant",
               "probiotic", "prebiotic", "omega", "nutrients", "cholesterol", "saturated", "sucrose",
               "dextrose", "glucose", "fructose", "lactose", "whey", "casein", "allergens"}
NUTRIENT_SHORT = {"dha", "epa", "ara", "vit", "vits", "zinc", "iron", "fibre", "fiber", "mufa", "pufa",
                  "gda", "rda", "kcal", "kj", "ins", "msg", "bcaa", "mct"}


def is_stop(text: str) -> bool:
    t = text.lower()
    if any(p in t for p in STOP_PHRASES):
        return True
    if any(norm(w) in NUTRIENT_SHORT for w in words(t)):
        return True
    # OCR variants of regulatory words: "Jssai" for fssai, "lngredients"
    return any(len(norm(w)) >= 5 and any(SequenceMatcher(None, norm(w), st).ratio() >= 0.8 for st in STOP_TOKENS)
               for w in words(t))


def is_generic(tok: str) -> bool:
    """Generic word, incl. OCR variants ("ENERG" ~ energy) and hyphenated compounds whose
    parts are all generic ("Karela-Amla", "Sugar-Free")."""
    k = norm(tok)
    if not k:
        return True
    if k in BRAND_GENERIC:
        return True
    parts = [norm(x) for x in re.split(r"[-/&+]", tok) if norm(x)]
    if len(parts) > 1 and all(x in BRAND_GENERIC for x in parts):
        return True
    if len(k) >= 5 and any(len(g) >= 5 and SequenceMatcher(None, k, g).ratio() >= 0.85 for g in GENERIC_LONG):
        return True
    return False


def in_dictionary(tok: str) -> bool:
    k = norm(tok)
    if k in DICT_WORDS:
        return True
    parts = [norm(x) for x in re.split(r"[-/]", tok) if norm(x)]
    return len(parts) > 1 and all(x in DICT_WORDS for x in parts)


def is_spec(tok: str) -> bool:
    k = tok.strip("(),.;:")
    parts = re.split(r"[-/]", k)
    if len(parts) > 1 and any(norm(p) in UNIT_LIKE or norm(p) in UNIT_WORDS for p in parts[1:]):
        return True
    if len(parts) >= 3 and all(p for p in parts):  # "MIL-STD-810H": standards, not models
        return True
    return bool(SPEC_TOKEN.match(k)) or norm(k) in UNIT_WORDS or bool(re.match(r"^i[3579]-?\d*", k, re.I))


def strip_generic(text: str) -> str:
    """Drop generic words from both ends of a line: 'NEW STING ENERGY' -> 'STING'."""
    ws = words(text)
    while ws and is_generic(ws[0]):
        ws.pop(0)
    while ws and is_generic(ws[-1]):
        ws.pop()
    return " ".join(ws)


def all_dictionary(text: str) -> bool:
    ws = [w for w in words(text) if norm(w)]
    return bool(DICT_WORDS) and bool(ws) and all(in_dictionary(w) for w in ws)


def brand_candidate(text: str) -> str | None:
    """The brand-shaped core of a line, or None if it cannot be a brand."""
    text = clean(text)
    if not text or alpha_ratio(text) < 0.6:
        return None
    if len(words(text)) > 4 or text.endswith((".", ",", ":")) or "(" in text or ":" in text or "," in text:
        return None
    raw_ws = words(text)
    if len(raw_ws) >= 2 and norm(raw_ws[-1]) in CONNECTORS:   # "Advantages of": a sentence cut mid-way
        return None
    # stop-list on the core, not the raw line: "Boost Nutrition Drink" -> "Boost" is fine,
    # "INGREDIENTS Water" -> "INGREDIENTS" is still caught
    core = " ".join(w for w in words(strip_generic(text)) if len(norm(w)) > 1 or len(words(text)) == 1)
    core = strip_generic(core)
    if not core or len(norm(core)) < 3 or is_stop(core):   # "So", "Af", "Tm": fragments
        return None
    cws = words(core)
    if all(is_generic(w) for w in cws):
        return None
    if re.fullmatch(r"[\W\d]+", core):
        return None
    if all(re.search(r"\d", w) for w in cws):  # "85dBA", "770NC": specs/models, not brands
        return None
    if any(norm(w) in CONNECTORS for w in cws[1:-1]):   # "Installed by HiDevolution", "Team Your Laptop"
        return None
    if len(cws) >= 2 and (norm(cws[0]) in CONNECTORS or norm(cws[-1]) in CONNECTORS):  # "productivity with"
        return None
    if len(cws) >= 3 and all(in_dictionary(w) for w in cws):  # "Eye-catching design and portable"
        return None
    if not re.search(r"[A-Za-z]", core):            # Cyrillic etc.: not our catalog's brands
        return None
    if len(cws) >= 2 and cws[0][0].islower() and not MIXED_CASE.match(cws[0]):  # "streamers alike": sentence fragment
        return None
    return core


def lower_dict_word(text: str) -> bool:
    """'performance', 'blue': a lowercase plain English word. Penalised, not excluded -
    'boat' (boAt) and 'vivo' are brands written in lowercase."""
    ws = words(text)
    return len(ws) == 1 and ws[0][0].islower() and in_dictionary(ws[0])


MIXED_CASE = re.compile(r"^(?:[a-z]+[A-Z][A-Za-z]*|[A-Z][a-z]+[A-Z][A-Za-z]*)$")


def mixed_case_candidates(text: str) -> list[str]:
    """'Level up with boAt and funk up' -> ['boAt']; 'PediaSure is a Registered...' -> ['PediaSure'].
    CamelCase / inner-capital tokens are almost always brand or product-line names."""
    out = []
    for w in words(clean(text)):
        w = w.strip("*:,.()®™'\"")
        # the casing itself is the signal, so no dictionary test ("boAt" is a word)
        if MIXED_CASE.match(w) and len(w) >= 4 and not is_generic(w):
            out.append(w)
    return out


def trademark_marked(text: str) -> list[str]:
    """Tokens carrying ® / ™ or named in 'X is a (registered) trademark' / 'product of X'."""
    out = []
    for m in re.finditer(r"([A-Za-z][A-Za-z'\-]{2,})\s?[®™]", text):
        out.append(m.group(1))
    m = re.search(r"^\s*([A-Za-z][A-Za-z'\-]{2,}(?:\s[A-Za-z][A-Za-z'\-]{2,})?)\s+is\s+a\s+(registered\s+)?trademark", text, re.I)
    if m:
        out.append(m.group(1))
    m = re.search(r"\b(?:a\s+)?product\s+of\s+([A-Z][A-Za-z'\-]{2,})", text)
    if m:
        out.append(m.group(1))
    return [w for w in out if not is_generic(w)]


def generic_phrase_candidate(text: str) -> str | None:
    """A 2-4 word line made only of generic words ("Made From Plants"). Excluded by
    brand_candidate; allowed as a weak candidate so an all-generic brand can still win
    when nothing brand-shaped recurs."""
    text = clean(text)
    ws = words(text)
    if not 3 <= len(ws) <= 4 or is_stop(text) or alpha_ratio(text) < 0.8 or len(norm(text)) < 6:
        return None   # 2-word generic pairs ("Water Resistance") are features, not brands
    if not all(is_generic(w) for w in ws) or any(norm(w) in UNIT_WORDS or norm(w) in {norm(x) for x in CO_BRANDS} for w in ws):
        return None
    if any(norm(w) in CONNECTORS for w in (ws[0], ws[-1])):
        return None
    return text


def first_word_candidate(text: str) -> str | None:
    """'JBL HEADPHONES APP', 'TAGRY Auriculares Bluetooth ...', 'ASUS Vivobook': the first
    word of a listing title / feature line is the brand by e-commerce convention, when it
    is an alphabetic non-dictionary token."""
    ws = words(clean(text))
    if len(ws) < 2 or len(ws) > 15 or is_stop(text) or "(" in text or text.rstrip().endswith("."):
        return None
    if ws[0].endswith(","):  # "Substances, Colour (110)..." is a list, not a title
        return None
    w = ws[0].strip("*:,.()")
    if not w.isalpha() or len(norm(w)) < 3 or (w[0].islower() and not w.isupper()):
        return None
    k = norm(w)
    if is_generic(w) or k in DICT_WORDS or is_stop(w):
        return None
    return w


# =============================================================================
# Line pool
# =============================================================================


def image_index(path: str) -> int:
    m = re.search(r"Image_(\d+)_", path)
    return int(m.group(1)) if m else 99


SEG_VOCAB = None


def segment_glued(tok: str) -> str:
    """Spotting sometimes glues words: 'MALTBASEDFOOD', 'SUPERVALUEPACK', 'ENERGYDRINK',
    'BoostNutrition'. Split an all-caps run (>= 9 letters) greedily into generic /
    dictionary words when the WHOLE token segments; split CamelCase when the trailing
    part is a generic word ('BoostNutrition' -> 'Boost Nutrition'). Brands that do not
    segment ('ZEBRONICS', 'PediaSure') are left alone."""
    global SEG_VOCAB
    if SEG_VOCAB is None:
        SEG_VOCAB = {w for w in BRAND_GENERIC if len(w) >= 3} | {w for w in DICT_WORDS if len(w) >= 4}
    m = re.match(r"^([A-Z][a-z]{3,})([A-Z][a-z]{3,})$", tok)
    if m and norm(m.group(2)) in BRAND_GENERIC:
        return f"{m.group(1)} {m.group(2)}"
    if not (tok.isupper() and tok.isalpha() and len(tok) >= 9):
        return tok
    low = tok.lower()

    def seg(i: int) -> list[str] | None:
        if i == len(low):
            return []
        for j in range(min(len(low), i + 14), i + 2, -1):
            if low[i:j] in SEG_VOCAB:
                rest = seg(j)
                if rest is not None:
                    return [low[i:j]] + rest
        return None

    parts = seg(0)
    return " ".join(p.upper() for p in parts) if parts and len(parts) >= 2 else tok


def repair_text(text: str) -> str:
    return " ".join(segment_glued(w) for w in words(text)) if text else text


def stacked_blocks(lines: list[dict]) -> list[dict]:
    """Logos set as 2-3 stacked short lines ('made' / 'from' / 'plants', 'O\'cean' / 'ONE8',
    'RiteBite' / 'MAX'): synthesize one merged line per run so the phrase can compete
    as a single candidate. Runs need x-overlap, a small gap, similar size, <= 2 words each."""
    out = []
    ordered = sorted(lines, key=lambda l: (l["box"][1], l["box"][0]))
    for i, a in enumerate(ordered):
        if len(words(a["text"])) > 2 or is_stop(a["text"]):
            continue
        run = [a]
        cur = a
        for b in ordered[i + 1:]:
            if len(words(b["text"])) > 2 or is_stop(b["text"]):
                continue
            gap = b["box"][1] - cur["box"][3]
            overlap = min(b["box"][2], cur["box"][2]) - max(b["box"][0], cur["box"][0])
            size_ok = min(a["size"], b["size"]) / max(a["size"], b["size"], 1) >= 0.45
            if -0.3 * cur["size"] <= gap <= 0.8 * cur["size"] and overlap > 0 and size_ok:
                run.append(b)
                cur = b
                if len(run) == 3:
                    break
            elif gap > 0.8 * cur["size"]:
                break
        if len(run) >= 2 and len({norm(l["text"]) for l in run}) == len(run):
            xs = [l["box"][0] for l in run] + [l["box"][2] for l in run]
            ys = [l["box"][1] for l in run] + [l["box"][3] for l in run]
            out.append({"text": " ".join(l["text"].strip() for l in run), "box": [min(xs), min(ys), max(xs), max(ys)],
                        "quad": None, "height": max(ys) - min(ys), "width": max(xs) - min(xs),
                        "conf": min((l["conf"] or 0) for l in run) or None,
                        "size": max(l["size"] for l in run), "block": True})
    return out


def content_key(rec: dict) -> str:
    """Order photos by what is ON them, never by upload position, so every tie-break in
    the rules is the same whichever photo the seller uploaded first."""
    if "lines" in rec:
        return "\n".join(sorted(l["text"] for l in rec["lines"])) + rec["image"].split("_", 2)[-1]
    return (rec.get("text") or "") + rec["image"].split("_", 2)[-1]


# Scales the "front-like photo" bonuses in score_brands, per category. Grocery packs have
# a real front panel; electronics listings are mostly feature infographics with no front,
# and there the bonus measurably hurt (58% -> 52% brand match), so it is off.
FRONT_WEIGHT = {"grocery": 1.0, "electronics": 0.0}


def front_likeness(lines: list[dict]) -> float:
    """0..1: how much this photo looks like a pack front / listing hero, from content only
    (never from upload position).

    A front has a few big, short lines that are names rather than claims. Things that are
    NOT fronts, and the evidence used against each:
      ingredient / nutrition / address panels  -> regulatory lines (stop_ratio)
      paragraphs of marketing copy             -> 6+ word lines (body_ratio)
      feature infographics ("AMBIENT AWARE" with a sentence under it, "120 Hours",
      "IPX4 Water Resistance")                 -> big lines that are headlines or carry digits
    Big short lines made only of plain English words ("PERFECT SOUND", "Smooth") count half:
    pack fronts carry invented words (ZANDU, Horlicks, Glucon-D)."""
    if not lines:
        return 0.0
    stop_ratio = sum(is_stop(l["text"]) for l in lines) / len(lines)
    body_ratio = sum(len(words(l["text"])) >= 6 for l in lines) / len(lines)
    big = [l for l in lines if l["rel_h"] >= 0.5 and not l.get("block")]
    if not big:
        return 0.0
    big_short = [l for l in big if len(words(l["text"])) <= 4 and alpha_ratio(l["text"]) >= 0.6
                 and not is_stop(l["text"])]
    if not big_short:
        return 0.0
    headline_ratio = sum(l.get("headline", False) for l in big) / len(big)
    digit_ratio = sum(bool(re.search(r"\d", l["text"])) for l in big) / len(big)
    name_like = sum(0.5 if all_dictionary(l["text"]) else 1.0 for l in big_short)
    score = ((1 - stop_ratio) * (1 - body_ratio) * (1 - headline_ratio) * (1 - 0.6 * digit_ratio)
             * min(1.0, name_like / 1.5))
    return round(score, 3)


def build_pool(spot_recs: list[dict], ocr_recs: list[dict]):
    """per_image: [(img_idx, [line dict + size/rel_h/rel_y])] from Spotting, ordered by
    content; ocr_lines: {img_idx: [plain OCR lines]}; all_text: {img_idx: normalised blob};
    image_info: {img_idx: {"front": 0..1, "front_like": bool, "regulatory": bool}}.
    img_idx is only an identifier - no rule may treat any index specially."""
    per_image = []
    for rec in sorted(spot_recs, key=content_key):
        lines = [dict(l) for l in rec["lines"] if l.get("box") and l["text"].strip()]
        for l in lines:  # font size: min(w, h) - a vertical logo's font size is its WIDTH
            l["text"] = repair_text(l["text"])
            l["size"] = min(l["width"], l["height"]) if len(l["text"].strip()) > 1 else l["height"]
        lines += stacked_blocks(lines)
        max_h = max((l["size"] for l in lines), default=1) or 1
        img = image_index(rec["image"])
        for l in lines:
            # Infographic headline: a short line with sentence-case body copy (3+ words,
            # lowercase letters present) directly under it. Brands never have body copy
            # hanging off them; feature callouts ("AMBIENT AWARE / AND TALKTHRU") always do.
            # Only lines made of ordinary words qualify: a logo with a tagline under it
            # (ZANDU / "Aloe Vera Health Juice", boAt / "Signature Sound") is not a callout.
            l["headline"] = len(words(l["text"])) <= 4 and all(
                in_dictionary(w) or is_generic(w) for w in words(l["text"]) if norm(w)
            ) and not any(MIXED_CASE.match(w) for w in words(l["text"])) and any(
                o is not l and len(words(o["text"])) >= 3
                and re.search(r"[a-z]", o["text"]) and not o["text"].isupper()
                and 0 <= o["box"][1] - l["box"][3] <= 2.0 * l["size"]
                and min(o["box"][2], l["box"][2]) > max(o["box"][0], l["box"][0])
                for o in lines)
        for l in lines:
            # Nutrition-table row: a value with a unit directly to the right or below
            # ("TAURINE" / "400 mg"). Never a brand.
            l["table_row"] = any(
                o is not l and re.fullmatch(r"[\d.,<>]+\s*(mg|g|mcg|kcal|ml|%|iu|µg|ug)?\s*(\(.*\))?", o["text"].strip(), re.I)
                and ((abs(o["box"][1] - l["box"][1]) <= l["size"] and o["box"][0] >= l["box"][2] - 5)
                     or (0 <= o["box"][1] - l["box"][3] <= 0.8 * l["size"]
                         and min(o["box"][2], l["box"][2]) > max(o["box"][0], l["box"][0])))
                for o in lines)
        stops = sum(is_stop(l["text"]) for l in lines)
        regulatory = bool(lines) and stops / len(lines) >= 0.5   # ingredient / nutrition close-up
        if not lines or stops / len(lines) < 0.3:               # a can front with "250 ml" is not a table
            for l in lines:
                l["table_row"] = False
        per_image.append((img, [{**l, "rel_h": l["size"] / max_h, "rel_y": l["box"][1] / 1000,
                                 "img": img, "regulatory": regulatory} for l in lines]))

    image_info = {}
    for img, lines in per_image:
        image_info[img] = {"front": front_likeness(lines),
                           "regulatory": bool(lines) and lines[0]["regulatory"]}
    # Front-like = clearly front-ish, or (if no photo is) every non-regulatory photo.
    best = max((i["front"] for i in image_info.values()), default=0.0)
    for img, info in image_info.items():
        info["front_like"] = (info["front"] >= 0.4 or (best < 0.4 and not info["regulatory"])) and info["front"] > 0
    if not any(i["front_like"] for i in image_info.values()):
        for info in image_info.values():
            info["front_like"] = not info["regulatory"]
    # Most front-like photos first (ties: content), so iteration order is content-driven.
    per_image.sort(key=lambda t: (-image_info[t[0]]["front"],))

    ocr_lines: dict[int, list[str]] = defaultdict(list)
    for rec in sorted(ocr_recs, key=content_key):
        for l in (rec.get("text") or "").splitlines():
            if l.strip():
                ocr_lines[image_index(rec["image"])].append(repair_text(l.strip()))

    all_text: dict[int, str] = defaultdict(str)
    for img, lines in per_image:
        all_text[img] += " " + " ".join(norm(l["text"]) for l in lines)
    for img, lines in ocr_lines.items():
        all_text[img] += " " + " ".join(norm(l) for l in lines)
    return per_image, ocr_lines, all_text, image_info


# =============================================================================
# Brand
# =============================================================================


def score_brands(per_image, ocr_lines, all_text, image_info, category: str = "Grocery") -> list[dict]:
    front_w = FRONT_WEIGHT.get("electronics" if category.lower().startswith("elec") else "grocery", 1.0)
    n_images = max(1, len(set(all_text) | {img for img, _ in per_image}))
    front_imgs = {img for img, i in image_info.items() if i["front_like"]}
    clusters: list[dict] = []

    def cluster_for(key: str) -> dict:
        for c in clusters:
            if similar(c["key"], key):
                return c
        c = {"key": key, "images": set(), "spellings": defaultdict(float), "best_rel_h": 0.0,
             "best_rel_y": 1.0, "front_hit": False, "min_words": 9, "title_hits": 0,
             "headline_hits": 0, "geo_hits": 0, "table_hits": 0, "tm_hits": 0,
             "generic_phrase": False, "largest_imgs": set()}
        clusters.append(c)
        return c

    def add_geo(c: dict, core: str, l: dict, img: int) -> None:
        c["images"].add(img)
        c["spellings"][core] += 1 + (l["conf"] or 0)
        c["best_rel_h"] = max(c["best_rel_h"], l["rel_h"])
        c["best_rel_y"] = min(c["best_rel_y"], l["rel_y"])
        c["front_hit"] |= img in front_imgs and l["rel_h"] >= 0.5
        c["min_words"] = min(c["min_words"], len(words(core)))
        c["headline_hits"] += int(l.get("headline", False))
        if l["rel_h"] >= 0.95 and not l.get("block"):
            c["largest_imgs"].add(img)   # THE biggest text on this photo
        c["table_hits"] += int(l.get("table_row", False))
        c["geo_hits"] += 1

    # Spotted lines found clusters (they carry geometry).
    for img, lines in per_image:
        for l in lines:
            core = brand_candidate(l["text"])
            if core:
                add_geo(cluster_for(norm(core)), core, l, img)
            else:
                gp = generic_phrase_candidate(l["text"])   # "Made From Plants": all-generic logo line
                if gp:
                    c = cluster_for(norm(gp))
                    add_geo(c, gp, l, img)
                    c["generic_phrase"] = True
            for w in mixed_case_candidates(l["text"]):
                if not core or norm(w) != norm(core):
                    c = cluster_for(norm(w))
                    c["images"].add(img)
                    c["spellings"][w] += 1
                    c["min_words"] = 1
            for w in trademark_marked(l["text"]):
                c = cluster_for(norm(w))
                c["images"].add(img)
                c["spellings"][w] += 0.5
                c["tm_hits"] += 1
                c["min_words"] = min(c["min_words"], len(words(w)))
            fw = first_word_candidate(l["text"])
            if fw and (not core or norm(fw) != norm(core)):
                c = cluster_for(norm(fw))   # recurrence + spelling only, no geometry
                c["images"].add(img)
                c["spellings"][fw] += 1
                c["min_words"] = min(c["min_words"], 1)
                if len(words(l["text"])) >= 5:
                    c["title_hits"] += 1

    # Plain-OCR lines: spelling + recurrence votes. Title lines may found a cluster.
    for img, lines in ocr_lines.items():
        for l in lines:
            for w in mixed_case_candidates(l):
                c = cluster_for(norm(w))
                c["images"].add(img)
                c["spellings"][w] += 1
                c["min_words"] = 1
            for w in trademark_marked(l):
                c = cluster_for(norm(w))
                c["images"].add(img)
                c["spellings"][w] += 0.5
                c["tm_hits"] += 1
                c["min_words"] = min(c["min_words"], len(words(w)))
            fw = first_word_candidate(l)
            if fw:
                c = cluster_for(norm(fw))
                c["images"].add(img)
                c["spellings"][fw] += 1.5
                c["min_words"] = min(c["min_words"], 1)
                if len(words(l)) >= 5:
                    c["title_hits"] += 1
            core = brand_candidate(l)
            if core:
                key = norm(core)
                for c in clusters:
                    if similar(c["key"], key):
                        c["images"].add(img)
                        c["spellings"][core] += 1.5  # "OCR:" spells stylised text better
                        break

    # Topmost large candidate on each front-like photo: brands sit above variant names.
    # Counted over ALL front-like photos, so no single photo decides it.
    topmost_counts: dict[str, int] = defaultdict(int)
    for img, lines in per_image:
        if img in front_imgs:
            big = [l for l in sorted(lines, key=lambda l: (l["box"][1], l["box"][0], l["text"]))
                   if l["rel_h"] >= 0.5 and brand_candidate(l["text"])]
            if big:
                topmost_counts[norm(brand_candidate(big[0]["text"]))] += 1

    regulatory_imgs = {img for img, i in image_info.items() if i["regulatory"]}
    tokens_by_img: dict[int, set[str]] = defaultdict(set)
    list_members: set[str] = set()      # tokens seen inside comma lists: ingredients / herbs
    for img, lines in per_image:
        for l in lines:
            tokens_by_img[img] |= {norm(w) for w in words(l["text"])}
            if l["text"].count(",") >= 2:
                list_members |= {norm(w) for w in words(l["text"].replace(",", " "))}
    for img, lines in ocr_lines.items():
        for l in lines:
            tokens_by_img[img] |= {norm(w) for w in words(l)}
            if l.count(",") >= 2:
                list_members |= {norm(w) for w in words(l.replace(",", " "))}

    # "ASUS Vivobook", "boAt Airdopes 500", "Apple iPhone": a candidate that is immediately
    # followed, on 2+ lines, by another candidate or a model token is the brand half.
    keys = [c["key"] for c in clusters]
    before_counts: dict[str, int] = defaultdict(int)
    texts = [l["text"] for _, lines in per_image for l in lines] + [l for ls in ocr_lines.values() for l in ls]
    for t in texts:
        ws = [w.strip("*:,.()®™") for w in words(clean(t))]
        for i in range(len(ws) - 1):
            a, b = norm(ws[i]), ws[i + 1]
            if len(a) < 3 or is_generic(ws[i]) or not re.search(r"[A-Za-z]", ws[i]):
                continue
            # the following token must look like a product line: a model token, a
            # CamelCase token, or a capitalised non-dictionary word (Vivobook, Airdopes)
            nxt_is_line = ((MODEL_TOKEN.match(b) and not is_spec(b)) or MIXED_CASE.match(b)
                           or (b[:1].isupper() and len(norm(b)) >= 4 and not in_dictionary(b)))
            if nxt_is_line and not is_generic(b) and norm(b) not in CONNECTORS:
                before_counts[a] += 1
    brand_before = {k for k, n in before_counts.items() if n >= 2}

    for c in clusters:
        c["spelling"] = max(c["spellings"], key=c["spellings"].get)
        c["images"] |= {img for img, t in all_text.items() if c["key"] in t}
        if len(c["key"]) >= 5:   # "Hakdirams" ~ "Haldiram's", "PATANJALL" ~ "Patanjali"
            c["images"] |= {img for img, toks in tokens_by_img.items()
                            if any(len(t) >= 5 and similar(t, c["key"]) for t in toks)}
        c["list_member"] = len(words(c["spelling"])) == 1 and c["key"] in list_members
        # Recurrence: an ingredient/nutrition close-up counts half - every nutrient name
        # recurs on those panels, the brand recurs on the sellable faces.
        rec_hits = sum(0.5 if img in regulatory_imgs else 1.0 for img in c["images"])
        recurrence = rec_hits / n_images
        acronym = (len(words(c["spelling"])) == 1 and c["spelling"].isupper()
                   and 2 <= len(c["spelling"]) <= 5 and norm(c["spelling"]) not in DICT_WORDS)
        c["score"] = round(
            2.0 * recurrence
            + 0.4 * (len(c["images"]) >= 2)          # seen on more than one photo at all
            + 0.6 * c["best_rel_h"]
            + 0.3 * (1 - c["best_rel_y"])
            + front_w * 0.2 * c["front_hit"]
            + front_w * 0.3 * (len(c["images"] & front_imgs) / max(1, len(front_imgs)))   # on the front-like photos
            + 0.2 * (c["min_words"] <= 2)
            + front_w * 0.3 * (sum(n for k, n in topmost_counts.items() if similar(c["key"], k)) / max(1, len(front_imgs)))
            + 0.3 * min(c["title_hits"], 2)          # leads a listing title
            + 0.4 * (c["tm_hits"] > 0 and (len(c["images"]) >= 2 or c["best_rel_h"] >= 0.5))  # ® / ™ on a logo or recurring
            + 0.6 * (c["key"] in brand_before)       # precedes a product line ("ASUS Vivobook", "boAt Airdopes")
            + 0.35 * min(len(c["largest_imgs"]), 2)  # biggest text on its photo (logo), any photo, any order
            + 0.4 * (any(MIXED_CASE.match(sp) and len(sp) >= 4 for sp in c["spellings"])   # CamelCase: boAt, iPhone
                     and not (c["geo_hits"] > 0 and c["headline_hits"] / c["geo_hits"] >= 0.5))
            - 0.5 * all_dictionary(c["spelling"])
            - 0.6 * (c["geo_hits"] > 0 and c["headline_hits"] / c["geo_hits"] >= 0.5)  # feature headline
            - 0.8 * (c["geo_hits"] > 0 and c["table_hits"] / c["geo_hits"] >= 0.5)     # nutrition row
            + 0.3 * (acronym and len(c["images"]) >= 2)                                 # JBL, ASUS, MSI
            - (0.6 if (c["best_rel_h"] >= 0.8 and (len(c["images"]) >= 2 or c["front_hit"])) else 1.5) * c["generic_phrase"]  # "made from plants" logo block
            - 0.8 * c["list_member"]                  # named inside a comma list: an ingredient
            - 0.8 * (lower_dict_word(c["spelling"]) and not any(MIXED_CASE.match(sp) for sp in c["spellings"]))  # "boosts", "harmful"
            - 1.0 * any(re.search(r"\d", w) for w in words(c["spelling"])), 3)  # "TUNE 770 NC" / "3X MORE": model or claim, not brand
    # final hygiene: a cluster whose chosen spelling is itself generic / regulatory is out
    clusters = [c for c in clusters if c["generic_phrase"]
                or not (all(is_generic(w) for w in words(c["spelling"])) or is_stop(c["spelling"]))]
    clusters.sort(key=lambda c: (-c["score"], c["key"]))   # ties: alphabetical, never upload order
    return clusters


def adjacent_pair(per_image, key_a: str, key_b: str) -> list[str] | None:
    """Logo made of two stacked large lines ("O'cean" / "ONE8")."""
    for _, lines in per_image:
        la = [l for l in lines if (c := brand_candidate(l["text"])) and similar(norm(c), key_a)]
        lb = [l for l in lines if (c := brand_candidate(l["text"])) and similar(norm(c), key_b)]
        for a in la:
            for b in lb:
                gap = max(a["box"][1], b["box"][1]) - min(a["box"][3], b["box"][3])
                if gap < 0.6 * max(a["size"], b["size"]) and a["rel_h"] >= 0.6 and b["rel_h"] >= 0.6:
                    # reading order: top-to-bottom, but left-to-right when either line is
                    # set vertically ("O'cean" printed sideways next to "ONE8")
                    rotated = any(l["height"] > 2 * l["width"] for l in (a, b))
                    first, second = sorted((a, b), key=lambda l: l["box"][0] if rotated else l["box"][1])
                    return [brand_candidate(first["text"]), brand_candidate(second["text"])]
    return None


def predecessor(scored, per_image, ocr_lines, key: str) -> dict | None:
    """'ASUS Vivobook', 'Apple iPhone', 'boAt Airdopes': when the winner is usually
    preceded on a line by another candidate, that candidate is the brand and the winner
    is the product line."""
    counts: dict[str, int] = defaultdict(int)
    texts = [l["text"] for _, lines in per_image for l in lines] + [l for ls in ocr_lines.values() for l in ls]
    for t in texts:
        ws = [w.strip("*:,.()®™") for w in words(clean(t))]
        for i in range(1, len(ws)):
            raw_prev = words(clean(t))[i - 1]
            if raw_prev.endswith((",", ";")):          # "LAVANGA, SUNTHI, YAVANI": a list, not brand + line
                continue
            if similar(norm(ws[i]), key) and ws[i - 1] and not is_generic(ws[i - 1]) and norm(ws[i - 1]) not in CONNECTORS:
                counts[norm(ws[i - 1])] += 1
    for pk, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if n >= 2:
            for c in scored:
                if similar(c["key"], pk) and not c["generic_phrase"]:
                    return c
    return None


def pick_brand(scored, per_image, ocr_lines=None) -> str | None:
    if not scored:
        return None
    pred = predecessor(scored, per_image, ocr_lines or {}, scored[0]["key"])
    if pred is not None and pred["score"] >= scored[0]["score"] - 1.2:
        scored.remove(pred)
        scored.insert(0, pred)
    brand = scored[0]["spelling"]
    ws = words(brand)
    if len(ws) == 2:
        first = next((c for c in scored[1:] if c["key"] == norm(ws[0])), None)
        if first and len(first["images"]) >= len(scored[0]["images"]):
            return first["spelling"]          # "ASUS Vivobook", "Horlicks Diabetes": brand + product line
    top = scored[0]
    if len(ws) != 1:                          # only two single-word logo lines are ever joined
        return brand
    for second in scored[1:4]:                # stacked two-line logo: "O'cean" / "ONE8"
        if len(words(second["spelling"])) != 1:
            continue
        if (second["best_rel_h"] >= 0.6 and top["best_rel_h"] >= 0.6
                and second["score"] >= top["score"] - 1.0   # physically stacked, so a wider gap is fine
                and not top["tm_hits"]):              # "V-Nourish®" / "PediaPlus": ® ends the brand
            pair = adjacent_pair(per_image, top["key"], second["key"])
            if pair and top["key"] not in second["key"] and second["key"] not in top["key"]:
                brand = " ".join(pair)
                break
    return brand


# =============================================================================
# Product name
# =============================================================================


def name_line_ok(t: str) -> bool:
    if not t or is_stop(t) or alpha_ratio(t) < 0.6 or len(words(t)) > 5:
        return False
    if t.endswith((".", ",")) or "(" in t:
        return False
    # loose units / numbers ("ml", "330", "556 mcg") are not name parts
    return not all(norm(w) in UNIT_WORDS or norm(w).isdigit() for w in words(t))


def collect_lines(lines, big_threshold, variants: bool) -> list[str]:
    ordered = sorted(lines, key=lambda l: (l["box"][1], l["box"][0]))
    keep: set[int] = set()
    for i, l in enumerate(ordered):
        t = clean(l["text"])
        if not name_line_ok(t):
            continue
        if big_threshold is not None and l["rel_h"] >= big_threshold:
            keep.add(i)
        elif variants and (VARIANT_RE.search(t) or VARIANT_SIZE.fullmatch(t) or is_flavour_line(t)):
            keep.add(i)
            # "vanilla" stacked right above "flavour", "NODED ORANGE" right below
            # "FLAVOUR": same block, keep the short adjacent neighbour too.
            for j in (i - 1, i + 1):
                if 0 <= j < len(ordered):
                    nb = ordered[j]
                    gap = max(l["box"][1], nb["box"][1]) - min(l["box"][3], nb["box"][3])
                    nt = clean(nb["text"])
                    if gap < max(nb["size"], l["size"]) and len(words(nt)) <= 2 and name_line_ok(nt):
                        keep.add(j)
    return [clean(ordered[i]["text"]) for i in sorted(keep)]


def dedupe_tokens(brand: str | None, parts: list[str], cap: int = 10) -> str | None:
    """Brand first, then the rest, deduped by normalised token; OCR variants of the brand
    ("Horlcks", "Glucan-D") are dropped by fuzzy match."""
    seen: set[str] = set()
    out: list[str] = []
    joined = ""
    for w in words(brand or "") + [w for p in parts for w in words(p)]:
        k = norm(w)
        if not k or k in seen or any(similar(k, sk) for sk in seen if len(k) >= 4):
            continue
        if len(k) >= 4 and k in joined:          # "Pedia" / "Plus" after "PediaPlus"
            continue
        if len(w) <= 3 and not (w.isupper() or w.islower() or w.istitle()):   # "dBu": OCR scrap
            continue
        w = w.rstrip("#*")
        seen.add(k)
        out.append(w)
        joined += k
    return pretty(" ".join(out[:cap])) if out else None


def grocery_name(brand, brand_key, per_image, ocr_lines, image_info) -> str | None:
    def brand_size_on(lines) -> float:
        return max((l["rel_h"] for l in lines
                    if (c := brand_candidate(l["text"])) and brand_key and similar(norm(c), brand_key)),
                   default=0.0)

    # Front = the front-like photo where the brand is printed biggest (ties: the more
    # front-like photo). Never "the first photo": sellers upload in any order. Back panels
    # that repeat the logo above a nutrition table are not front-like, so they lose.
    sizes = {img: brand_size_on(lines) for img, lines in per_image}
    pool = [img for img in sizes if image_info[img]["front_like"] and sizes[img] > 0] or list(sizes)
    front = max(pool, key=lambda i: (sizes[i] >= 0.3, image_info[i]["front"], sizes[i]), default=None)

    parts: list[str] = []
    for img, lines in per_image:
        if img == front:
            parts += collect_lines(lines, 0.3, variants=True)
    # "Protinex Original", "Boost Chocolate": the variant word printed right after the brand
    if brand_key:
        for img, lines in per_image:
            for l in lines:
                ws = [w.strip("*:,.()®™") for w in words(clean(l["text"]))]
                for i, w in enumerate(ws[:-1]):
                    if similar(norm(w), brand_key) and norm(ws[i + 1]) in FLAVOUR_WORDS:
                        parts.append(ws[i + 1])
    # Flavour/variant lines from the other front-like panels: the brand is on them and
    # they are not mostly regulatory text (ingredient / nutrition close-ups).
    def front_like(img, lines) -> bool:
        stops = sum(is_stop(l["text"]) for l in lines)
        return sizes.get(img, 0) > 0 and lines and stops / len(lines) < 0.5

    for img, lines in per_image:
        if img != front and front_like(img, lines):
            parts += collect_lines(lines, None, variants=True)
    if len({norm(w) for p in parts for w in words(p)}) < 3:  # bad boxes on the front
        for img, lines in per_image:
            if img != front and sizes.get(img, 0) >= 0.5:
                parts += collect_lines(lines, 0.5, variants=False)
    if len({norm(w) for p in parts for w in words(p)}) < 3 and brand_key:
        # plain "OCR:" reading order of the front: the short lines right after the brand
        front_ocr = ocr_lines.get(front, [])
        for i, l in enumerate(front_ocr):
            if brand_key in norm(l) or any(similar(norm(w), brand_key) for w in words(l)):
                for nxt in front_ocr[i + 1:i + 4]:
                    t = clean(nxt)
                    if len(words(t)) <= 3 and name_line_ok(t):
                        parts.append(t)
                break
    return dedupe_tokens(brand, parts)


def model_phrase(line: str, brand_key: str | None) -> str | None:
    """'JBL Tune 770NC Wireless Over-Ear...' -> 'Tune 770NC'; 'vivo Y28s 5G (Vintage Red...'
    -> 'Y28s 5G'. The phrase = up to two words before the first model token (not brand,
    not generic) + the model token + following model-ish tokens, stopping at a spec,
    a generic word or punctuation."""
    ws = [w.strip("(),;:") for w in words(clean(line))]
    ws = [w for w in ws if w]

    def followed_by_unit(i: int) -> bool:
        return i + 1 < len(ws) and norm(ws[i + 1]) in UNIT_LIKE | UNIT_WORDS

    def line_name_before(i: int) -> bool:
        if i == 0:
            return False
        w = ws[i - 1]
        k = norm(w)
        return (w[0].isalpha() and w[0].isupper() and k not in BRAND_GENERIC and k not in COLOUR_WORDS
                and not is_spec(w) and not (brand_key and similar(k, brand_key)))

    idx = None
    for i, w in enumerate(ws):
        k = norm(w)
        if k in BRAND_GENERIC or k in CO_BRANDS or is_spec(w) or followed_by_unit(i):
            continue
        if re.fullmatch(r"(19|20)\d\d", w) or (i > 0 and norm(ws[i - 1]) in MONTHS):
            continue
        if MODEL_TOKEN.match(w) and not re.match(r"^\d+(\.\d+)?[a-z]{1,2}$", w, re.I):
            idx = i
            break
        if BARE_NUMBER.match(w) and line_name_before(i):
            idx = i
            break
    if idx is None:
        return None
    # left context: product line name ("Tune", "Nord", "Airdopes", "Vivobook S")
    left: list[str] = []
    for w in reversed(ws[max(0, idx - 2):idx]):
        k = norm(w)
        if (not k or (brand_key and similar(k, brand_key)) or k in BRAND_GENERIC or k in COLOUR_WORDS
                or is_spec(w) or not w[0].isalpha() or (k in DICT_WORDS and not w[0].isupper())):
            break
        left.insert(0, w)
    # right context: more model tokens / suffixes ("Pro", "5G", "Plus", "NC")
    right: list[str] = []
    for w in ws[idx + 1:idx + 4]:
        k = norm(w)
        if (is_spec(w) or is_stop(w) or k in {"with", "and", "w"} or k in COLOUR_WORDS or k in UNIT_LIKE
                or (k in BRAND_GENERIC and k not in {"pro", "plus", "max", "5g", "4g", "lite", "mini", "oled", "s", "ultra"})):
            break
        right.append(w)
        if len(words(" ".join(right))) >= 3:
            break
    phrase = " ".join(left + [ws[idx]] + right)
    if alpha_ratio(phrase) < 0.4 or len(norm(phrase)) < 2:
        return None
    return phrase


def after_brand_phrase(line: str, brand_key: str) -> str | None:
    ws = [w.strip("(),;:") for w in words(clean(line))]
    ws = [w for w in ws if w]
    idx = next((i for i, w in enumerate(ws) if similar(norm(w), brand_key)), None)
    if idx is None:
        return None
    out: list[str] = []
    for w in ws[idx + 1:idx + 4]:
        k = norm(w)
        if (not k or k in BRAND_GENERIC or k in COLOUR_WORDS or is_spec(w) or is_stop(w)
                or k in UNIT_LIKE or not w[0].isalnum()):
            break
        if w[0].isalpha() and not w[0].isupper() and k in DICT_WORDS:
            break
        if k in DICT_WORDS and not re.search(r"\d", w) and len(out) == 0 and w.isupper() is False and w.istitle():
            # plain capitalised English word right after the brand ("Sony Headphones") is not a line name
            break
        out.append(w)
    return " ".join(out) if out else None


def electronics_name(brand, brand_key, per_image, ocr_lines, all_text) -> str | None:
    """Brand + the model phrase that recurs the most across images (ties: bigger print,
    nearer the top). Falls back to the grocery strategy when nothing model-like is read."""
    cands: dict[str, dict] = {}

    def vote(phrase: str, img: int, rel_h: float, rel_y: float, w: float = 1.0):
        key = norm(phrase)
        c = cands.setdefault(key, {"spellings": defaultdict(float), "images": set(), "rel_h": 0.0, "rel_y": 1.0})
        c["spellings"][phrase] += w
        c["images"].add(img)
        c["rel_h"] = max(c["rel_h"], rel_h)
        c["rel_y"] = min(c["rel_y"], rel_y)

    for img, lines in per_image:
        for l in lines:
            ph = model_phrase(l["text"], brand_key)
            if ph:
                vote(ph, img, l["rel_h"], l["rel_y"])
    for img, lines in ocr_lines.items():
        for l in lines:
            ph = model_phrase(l, brand_key)
            if ph:
                vote(ph, img, 0.0, 1.0, 0.8)

    # "ASUS Vivobook", "boAt Airdopes": the capitalised non-generic words right after the
    # brand on a line form the product line even when no digits were read.
    if brand_key:
        for img, lines in per_image:
            for l in lines:
                ph = after_brand_phrase(l["text"], brand_key)
                if ph:
                    vote(ph, img, l["rel_h"], l["rel_y"], 0.7)
        for img, lines in ocr_lines.items():
            for l in lines:
                ph = after_brand_phrase(l, brand_key)
                if ph:
                    vote(ph, img, 0.0, 1.0, 0.5)
    if not cands:  # nothing model-like read: the brand alone beats a list of feature callouts
        return dedupe_tokens(brand, [])

    n_images = max(1, len(all_text))
    for key, c in cands.items():
        c["images"] |= {img for img, t in all_text.items() if key in t}
        has_line_name = len(words(max(c["spellings"], key=c["spellings"].get))) >= 2
        c["score"] = (2.0 * len(c["images"]) / n_images + 0.6 * c["rel_h"] + 0.3 * (1 - c["rel_y"])
                      + 0.4 * has_line_name + 0.1 * min(sum(c["spellings"].values()), 5))
    best = max(cands.values(), key=lambda c: c["score"])
    phrase = max(best["spellings"], key=best["spellings"].get)
    electronics_name.last_debug = sorted(
        ({"phrase": max(c["spellings"], key=c["spellings"].get), "score": round(c["score"], 2),
          "images": sorted(c["images"])} for c in cands.values()), key=lambda d: -d["score"])[:5]
    return dedupe_tokens(brand, [phrase], cap=8)


# =============================================================================
# Driver
# =============================================================================


def extract(spot_recs, ocr_recs, category: str = "Grocery", debug: bool = False) -> dict:
    per_image, ocr_lines, all_text, image_info = build_pool(spot_recs, ocr_recs)
    scored = score_brands(per_image, ocr_lines, all_text, image_info, category)
    brand = pick_brand(scored, per_image, ocr_lines)
    brand_key = norm(brand) if brand else None
    if brand_key and len(words(brand)) == 2:  # stacked logo pair: match on either half
        brand_key = scored[0]["key"]

    electronics_name.last_debug = None
    if category.lower().startswith("elec"):
        name = electronics_name(brand, brand_key, per_image, ocr_lines, all_text)
    else:
        name = grocery_name(brand, brand_key, per_image, ocr_lines, image_info)

    result = {"brand": pretty(brand) if brand else None, "product_name": name}
    if debug:
        result["debug"] = {"name_candidates": getattr(electronics_name, "last_debug", None), "candidates": [
            {"spelling": c["spelling"], "score": c["score"], "images": sorted(c["images"]),
             "rel_h": round(c["best_rel_h"], 2)} for c in scored[:6]]}
    return result
