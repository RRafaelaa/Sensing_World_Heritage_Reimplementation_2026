import os
import re
import warnings
import pdfplumber
import pymupdf  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from collections import Counter

from pdf_to_text import extract_ocr

# Silence PyMuPDF warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

PDF_FOLDER = "data/pdfs"
TEXT_FOLDER = "data/texts"
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)


# ============================================================
# 0. SUPPRESS STDERR (PyMuPDF noise)
# ============================================================
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        yield


# ============================================================
# 1. PER-PAGE COLUMN DETECTION
# ============================================================
def detect_page_layout(words, page_width, threshold_ratio=0.38):
    if not words:
        return "single"

    xs = [float(w["x0"]) for w in words]
    median_x = sorted(xs)[len(xs)//2]
    return "double" if median_x < page_width * threshold_ratio else "single"


# ============================================================
# 2. ADAPTIVE HEADER + FOOTER REMOVAL
# ============================================================
def detect_repeating_header_footer(doc):
    header_counter = Counter()
    footer_counter = Counter()
    total_pages = len(doc)

    for page in doc:
        page_height = page.rect.height
        top_thresh = page_height * 0.08
        bottom_thresh = page_height * 0.92

        blocks = page.get_text("blocks")

        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            t = text.strip()
            if not t:
                continue

            # Normalize whitespace
            t_norm = " ".join(t.split())

            # Top of page
            if y0 < top_thresh:
                header_counter[t_norm] += 1

            # Bottom of page
            if y1 > bottom_thresh:
                footer_counter[t_norm] += 1

    # Keep only repeating lines
    min_repetition = max(2, total_pages * 0.3)

    repeating_headers = {t for t, c in header_counter.items() if c >= min_repetition}
    repeating_footers = {t for t, c in footer_counter.items() if c >= min_repetition}

    return repeating_headers, repeating_footers

def clean_blocks_of_headers_and_footers(blocks, repeating_headers, repeating_footers):
    """
    Remove any block whose normalized text matches a repeating header/footer line.
    """
    cleaned = []
    for b in blocks:
        x0, y0, x1, y1, text, *_ = b
        s_norm = " ".join(text.strip().split())

        if s_norm in repeating_headers or s_norm in repeating_footers:
            continue

        cleaned.append(b)

    return cleaned

def remove_header_footer_blocks(blocks, repeating_headers, repeating_footers, page_height):
    cleaned = []
    for b in blocks:
        x0, y0, x1, y1, text, *_ = b
        t = text.strip()
        t_norm = " ".join(t.split())

        # Remove if identified as repeating header/footer
        if t_norm in repeating_headers or t_norm in repeating_footers:
            continue

        cleaned.append(b)

    return cleaned

# ============================================================
# 3. PAGE EXTRACTION (SINGLE / DOUBLE)
# ============================================================
def extract_single_column_page(page, repeating_headers, repeating_footers):
    """
    Extract a single-column page while:
    - clipping top/bottom visual margins (generic removal)
    - removing repeating header/footer lines (structural removal)
    - preserving narrative text as cleanly as possible
    """

    header_clip = page.rect.height * 0.08
    footer_clip = page.rect.height * 0.08

    # 1) Extract text from the main content area ONLY (visual clipping)
    clip_rect = pymupdf.Rect(
        0,
        header_clip,
        page.rect.width,
        page.rect.height - footer_clip
    )

    # Raw clipped text (good for fallback)
    with suppress_stderr():
        raw_text = page.get_text("text", clip=clip_rect)

    # 2) Extract blocks WITHIN the clipped region — needed for filtering
    with suppress_stderr():
        all_blocks = page.get_text("blocks")

    # Filter blocks by clipping rectangle
    inner_blocks = []
    for b in all_blocks:
        x0, y0, x1, y1, text, *_ = b

        # Keep only blocks inside clipped region
        if y0 >= header_clip and y1 <= page.rect.height - footer_clip:
            inner_blocks.append(b)

    # 3) Remove repeating header/footer blocks
    cleaned_blocks = []
    for b in inner_blocks:
        x0, y0, x1, y1, text, *_ = b
        t_norm = " ".join(text.strip().split())

        if t_norm in repeating_headers or t_norm in repeating_footers:
            continue  # remove the header/footer text entirely

        cleaned_blocks.append(b)

    # 4) Sort blocks top-to-bottom, left-to-right
    cleaned_blocks = sorted(cleaned_blocks, key=lambda b: (b[1], b[0]))

    # 5) Join the cleaned block text
    cleaned_text = "\n".join(b[4].rstrip() for b in cleaned_blocks)

    # 6) Fallback: if block extraction fails, return clipped raw text
    return cleaned_text if cleaned_text.strip() else raw_text

def extract_double_column_page(page, repeating_headers, repeating_footers):
    with suppress_stderr():
        blocks = page.get_text("blocks")

    blocks = clean_blocks_of_headers_and_footers(
        blocks,
        repeating_headers,
        repeating_footers
    )
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

    txt = ""
    for b in blocks:
        txt += b[4] + "\n"
    return txt


# ============================================================
# 4. FIGURE CAPTIONS (ISOLATED)
# ============================================================
def extract_figure_captions(page, page_num):
    with suppress_stderr():
        raw = page.get_text("text")

    captions = []
    to_remove = []

    patterns = [
        r"(Figure\s*\d+[^.\n]*\.*)",
        r"(Fig\.\s*\d+[^.\n]*\.*)",
    ]

    for p in patterns:
        for m in re.finditer(p, raw, flags=re.IGNORECASE):
            cap_raw = m.group(1)
            to_remove.append(cap_raw)

            cap = cap_raw.replace("\u200b", " ")
            cap = re.sub(r"([a-z])([A-Z])", r"\1 \2", cap)
            cap = re.sub(r"(\d)([A-Za-z])", r"\1 \2", cap)
            cap = " ".join(cap.split())

            captions.append(f"[Figure p.{page_num}] {cap}")

    return captions, to_remove


# ============================================================
# 5. TABLE EXTRACTION (ISOLATED)
# ============================================================
def extract_tables_plumber(page, page_num):
    tables_clean = []
    try:
        tables = page.extract_tables()
        for ti, t in enumerate(tables, start=1):
            rows = [" | ".join([c or "" for c in row]) for row in t]
            tables_clean.append(f"[Table p.{page_num}, t{ti}]\n" + "\n".join(rows))
    except:
        pass
    return tables_clean


# ============================================================
# 6. PARAGRAPH RECONSTRUCTION
# ============================================================
def fix_hyphenation(text):
    return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)


def merge_inline_breaks(text):
    text = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', text)
    text = re.sub(r'([;:])\n(\S)', r'\1 \2', text)
    text = re.sub(r'\)\n([a-z])', r') \1', text)
    return text


def is_heading_like(text):
    if len(text.split()) <= 12 and (text.isupper() or text.istitle()):
        return True
    return False


def merge_short_wrapped_lines(text, threshold=45):
    lines = text.splitlines()
    merged = []
    buf = ""

    for line in lines:
        s = line.strip()

        if is_heading_like(s):
            if buf:
                merged.append(buf.strip())
                buf = ""
            merged.append(s)
            continue

        if len(s) < threshold and not s.endswith("."):
            buf += " " + s
        else:
            if buf:
                merged.append(buf.strip())
                buf = ""
            merged.append(s)

    if buf:
        merged.append(buf.strip())

    return "\n".join(merged)

MATH_OPERATORS = r'=+\-−–×*/<>≈≃→≥≤^'
MATH_SYMBOLS = '∑∏∫√∞πσμλΩωαβγδθϕφψηξτ'


def is_math_like(line: str) -> bool:
    """
    Heuristic: decide if a line is mostly math.
    No document-specific assumptions.
    """
    s = line.strip()
    if not s:
        return False

    # Contains a typical math operator?
    if re.search(f'[{MATH_OPERATORS}]', s):
        return True

    # Many non-alnum characters → likely math
    non_alnum = sum(1 for c in s if not (c.isalnum() or c.isspace()))
    if non_alnum / max(len(s), 1) > 0.35:
        return True

    # Contains known math symbols
    if any(ch in MATH_SYMBOLS for ch in s):
        return True

    return False


def fix_equations(text: str) -> str:
    """
    Merge broken multi-line equations into single lines.
    Treat consecutive math-like lines as one block.
    """
    lines = text.splitlines()
    out = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if is_math_like(line):
            block = [line.strip()]
            i += 1
            while i < n:
                nxt = lines[i]
                s = nxt.strip()
                if s == "":
                    i += 1
                    continue
                if is_math_like(nxt):
                    block.append(s)
                    i += 1
                else:
                    break

            eq = " ".join(block)

            # normalize spaces around operators
            eq = re.sub(rf'\s*([{MATH_OPERATORS}])\s*', r' \1 ', eq)
            eq = re.sub(r'\s+', ' ', eq).strip()

            # tighten parentheses/brackets
            eq = re.sub(r'\(\s+', '(', eq)
            eq = re.sub(r'\s+\)', ')', eq)
            eq = re.sub(r'\[\s+', '[', eq)
            eq = re.sub(r'\s+\]', ']', eq)

            out.append(eq)
        else:
            out.append(line)
            i += 1

    return "\n".join(out)

def normalize_paragraph_spacing(text):
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def reconstruct_paragraphs(text: str) -> str:
    """
    General paragraph reconstruction for academic PDFs.

    Handles (in a document-agnostic way):
    - hyphenation across line breaks
    - inline sentence breaks
    - citations in brackets [1, 2], [3–5]
    - author 'et al.' line breaks
    - numeric breaks ("The\\n3\\nlimitations")
    - broken parentheses and quotes
    - CamelCase / proper noun breaks ("WordCloud\\nPython")
    - section number references ("Section 2.2\\n," → "Section 2.2,")
    - equation blocks (via fix_equations)
    - heading preservation + short-line merging
    """

    # ---------------------------------------------------------
    # 1) HYPHENATION FIXES
    # ---------------------------------------------------------
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)


    # ---------------------------------------------------------
    # 2) PARENTHESIS FIXES (generic)
    # ---------------------------------------------------------
    # word\n( → word (
    text = re.sub(r'(\w)\s*\n\s*\(', r'\1 (', text)
    # (\n text \n) → (text)
    text = re.sub(r'\(\s*\n\s*([^)]+)\s*\n\s*\)', r'(\1)', text)


    # ---------------------------------------------------------
    # 3) QUOTE FIXES (generic)
    # ---------------------------------------------------------
    # “\ntext\n” → “text”
    text = re.sub(r'“\s*\n\s*([^”]+)\n\s*”', r'“\1”', text)
    # “\ntext → “text
    text = re.sub(r'“\s*\n\s*', '“', text)
    # text\n” → text”
    text = re.sub(r'\n\s*”', '”', text)


    # ---------------------------------------------------------
    # 4) INLINE CONTINUATION MERGES
    # ---------------------------------------------------------
    # lowercase\nlowercase → lowercase lowercase
    text = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1 \2', text)
    # comma/semicolon/colon/paren → continuation
    text = re.sub(r'([,;:])\s*\n\s*([^\s])', r'\1 \2', text)
    # .\nlowercase → . lowercase
    text = re.sub(r'\.\s*\n\s*([a-z])', r'. \1', text)
    # )\n( → ) (
    text = re.sub(r'\)\s*\n\s*\(', r') (', text)


    # ---------------------------------------------------------
    # 5) AUTHOR + "et al." (generic)
    # ---------------------------------------------------------
    text = re.sub(
        r'\b([A-Z][a-z]+,\s*[A-Z]\.)\s*\n\s*(et al\.)',
        r'\1 \2',
        text
    )


    # ---------------------------------------------------------
    # 6) NUMERIC BREAK FIXES ("The\\n3\\nlimitations")
    # ---------------------------------------------------------
    text = re.sub(r'(\w)\s*\n\s*(\d+)\s*\n\s*(\w)', r'\1 \2 \3', text)
    text = re.sub(r'(\w)\s*\n\s*(\d+)', r'\1 \2', text)
    text = re.sub(r'(\d+)\s*\n\s*(\w)', r'\1 \2', text)


    # ---------------------------------------------------------
    # 7) CITATION FIXES (generic bracketed)
    # ---------------------------------------------------------
    # [5 \n ] → [5]
    text = re.sub(r'\[\s*(\d+)\s*\n\s*\]', r'[\1]', text)
    # [1 , 2 \n ] → [1, 2]
    text = re.sub(r'\[\s*(\d+)\s*,\s*(\d+)\s*\n\s*\]', r'[\1, \2]', text)
    # [3 – 5 \n ] → [3–5]
    text = re.sub(r'\[\s*(\d+)\s*[-–]\s*(\d+)\s*\n\s*\]', r'[\1–\2]', text)
    # [\n ... \n] → [ ... ]
    text = re.sub(r'\[\s*\n([^]]+)\n\s*\]', r'[\1]', text)
    # ]\n[ → ], [
    text = re.sub(r'\]\s*\n\s*\[', r'], [', text)

    def _clean_cite(m):
        inside = m.group(1)
        inside = re.sub(r'\s*,\s*', ', ', inside)
        inside = re.sub(r'\s*[-–]\s*', '–', inside)
        return f'[{inside}]'

    text = re.sub(r'\[([^\]]+)\]', _clean_cite, text)


    # ---------------------------------------------------------
    # 8) SECTION REFERENCE FIXES (generic)
    # ---------------------------------------------------------
    # "Section 2.2\n," → "Section 2.2,"
    text = re.sub(
        r'(Section[s]?\s+\d+(\.\d+)*)\s*\n\s*([,.;)])',
        r'\1\3',
        text,
        flags=re.IGNORECASE
    )


    # ---------------------------------------------------------
    # 9) CAMELCASE / PROPER NOUN MERGES (generic)
    # ---------------------------------------------------------
    # "... using the\nWordCloud" → "using the WordCloud"
    text = re.sub(
        r'([a-z0-9,;])\s*\n\s*([A-Z][a-zA-Z0-9]+)',
        r'\1 \2',
        text
    )
    # "WordCloud\nlibrary" → "WordCloud library"
    text = re.sub(
        r'([A-Z][a-zA-Z0-9]+)\s*\n\s*([a-z])',
        r'\1 \2',
        text
    )


    # ---------------------------------------------------------
    # 10) GENERIC BROKEN-WORD MERGE
    #     (Keep this AFTER the specific patterns above)
    # ---------------------------------------------------------
    text = re.sub(
        r'([A-Za-z])\s*\n\s*([A-Za-z])',
        r'\1\2',
        text
    )


    # ---------------------------------------------------------
    # 11) EQUATION BLOCK MERGING
    # ---------------------------------------------------------
    text = fix_equations(text)


    # ---------------------------------------------------------
    # 12) REAL SECTION HEADING DETECTION
    #     (convert "... 1. Introduction" → "...\\n\\nSECTION: Introduction")
    # ---------------------------------------------------------
    HEADING_TERMS = [
        "Introduction", "Background", "Method", "Methods",
        "Materials and Methods", "Results", "Discussion",
        "Conclusion", "Conclusions"
    ]
    HEADING_REGEX = "(?:" + "|".join(
        term.replace(" ", r"\s+") for term in HEADING_TERMS
    ) + ")"

    # Two capture groups:
    # 1) preceding character
    # 2) heading text
    pattern = rf'(\w)\s+(?:\d+\.\s*)?({HEADING_REGEX})'

    def _fix_headings(match):
        prev, heading = match.groups()
        # if heading somehow starts lowercase, treat as normal text
        if not heading[0].isupper():
            return prev + " " + heading
        return prev + "\n\nSECTION: " + heading

    text = re.sub(pattern, _fix_headings, text, flags=re.IGNORECASE)


    # ---------------------------------------------------------
    # 13) FINAL SHORT-LINE MERGING (PARAGRAPH RECONSTRUCTION)
    # ---------------------------------------------------------
    merged_lines = []
    buf = ""

    def is_heading_like(line: str) -> bool:
        # Keep SECTION lines and ALLCAPS lines separate
        return (
            line.startswith("SECTION:") or
            line.isupper()
        )

    for line in text.splitlines():
        s = line.strip()

        # preserve blank lines as paragraph separators
        if not s:
            if buf:
                merged_lines.append(buf.strip())
                buf = ""
            merged_lines.append("")
            continue

        if is_heading_like(s):
            if buf:
                merged_lines.append(buf.strip())
                buf = ""
            merged_lines.append(s)
            continue

        # Merge "short" lines that don't end a sentence
        if len(s) < 70 and not s.endswith((".", "?", "!", ":")):
            buf += " " + s
        else:
            if buf:
                merged_lines.append(buf.strip())
                buf = ""
            merged_lines.append(s)

    if buf:
        merged_lines.append(buf.strip())

    text = "\n".join(merged_lines)

    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# ============================================================
# 7. FONT-AWARE BLOCK EXTRACTION
# ============================================================
def extract_blocks_with_fonts(page):
    with suppress_stderr():
        d = page.get_text("dict")

    blocks = []
    for b in d.get("blocks", []):
        for ln in b.get("lines", []):
            y0 = ln.get("bbox", [0,0,0,0])[1]
            for sp in ln.get("spans", []):
                txt = sp.get("text", "").strip()
                if not txt:
                    continue
                blocks.append({
                    "text": txt,
                    "size": sp.get("size", 0),
                    "bold": bool(sp.get("flags", 0) & 2),
                    "y0": y0,
                })
    return blocks


# ============================================================
# 8. HEADING DETECTION (FONT + REGEX)
# ============================================================
HEADING_PATTERN = re.compile(
    r"^(abstract|introduction|background|related work|methods?|materials and methods|"
    r"results?|discussion|conclusions?|limitations|future work|analysis)$",
    re.IGNORECASE
)


def detect_font_headings(blocks):
    if not blocks:
        return []
    sizes = sorted([b["size"] for b in blocks])
    body = sizes[len(sizes)//2] if sizes else 10

    candidates = []
    for b in blocks:
        w = len(b["text"].split())
        large = b["size"] >= body * 1.15
        bold_short = b["bold"] and w <= 12
        allcaps = b["text"].isupper() and 2 <= w <= 12
        if large or bold_short or allcaps:
            candidates.append(b)
    return candidates


def refine_headings(headings):
    refined = []
    for h in headings:
        t = h["text"]
        if re.match(r"^(fig|figure|table)\s*\d+", t, re.IGNORECASE):
            continue
        if HEADING_PATTERN.match(t):
            refined.append({**h, "name": t.title()})
        else:
            refined.append({**h, "name": t})
    return refined


def merge_multiline_headings(headings):
    merged = []
    headings = sorted(headings, key=lambda h: h["y0"])
    buf = None
    for h in headings:
        if buf is None:
            buf = h
        else:
            if abs(h["y0"] - buf["y0"]) < 20:
                buf["name"] += " " + h["name"]
            else:
                merged.append(buf)
                buf = h
    if buf:
        merged.append(buf)

    for h in merged:
        h["name"] = " ".join(w.capitalize() for w in h["name"].split())

    return merged


def split_by_headings(blocks, headings):
    headings = sorted(headings, key=lambda h: h["y0"])
    sections = {}

    cur = "Before First Heading"
    cur_text = []

    it = iter(headings)
    try:
        next_h = next(it)
    except StopIteration:
        sections["Body"] = "\n".join([b["text"] for b in blocks])
        return sections

    for b in blocks:
        if b is next_h:
            sections[cur] = "\n".join(cur_text).strip()
            cur = next_h["name"]
            cur_text = []
            try:
                next_h = next(it)
            except StopIteration:
                next_h = None
        else:
            cur_text.append(b["text"])

    sections[cur] = "\n".join(cur_text).strip()
    return sections


# ============================================================
# 9. FINAL ACADEMIC FORMATTING (NOW IN THIS FILE)
# ============================================================
def format_paragraphs_1blank(text):
    lines = [l.rstrip() for l in text.split("\n")]
    out = []
    prev_blank = False

    for l in lines:
        if not l.strip():
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(l)
            prev_blank = False

    return "\n".join(out).strip()


def assemble_layout(title, abstract, sections, tables, captions):
    parts = []

    if title:
        parts.append(f"TITLE:\n{title.strip()}")

    if abstract:
        abstract_fmt = format_paragraphs_1blank(reconstruct_paragraphs(abstract))
        parts.append(f"ABSTRACT:\n{abstract_fmt}")

    # Sections
    for name, content in sections.items():
        content_fmt = format_paragraphs_1blank(reconstruct_paragraphs(content))
        parts.append(f"SECTION: {name}\n{content_fmt}")

    # Tables
    if tables:
        parts.append("TABLES:\n" + "\n\n".join(tables))

    # Captions
    if captions:
        parts.append("FIGURE_CAPTIONS:\n" + "\n\n".join(captions))

    # TWO blank lines between major blocks
    return "\n\n\n".join(parts).strip()


# ============================================================
# 10. MAIN EXTRACTION PIPELINE
# ============================================================
def extract_text_from_pdf(pdf_path):
    all_tables = []
    all_captions = []
    blocks_all = []
    body_pages = []

    # PASS 1 – pdfplumber (layout detection + tables)
    try:
        with pdfplumber.open(pdf_path) as pdf_pl:
            layouts = []
            for i, page in enumerate(pdf_pl.pages):
                words = page.extract_words()
                layout = detect_page_layout(words, page.width)
                layouts.append(layout)
                all_tables.extend(extract_tables_plumber(page, i+1))
    except:
        layouts = []

    # PASS 2 – PyMuPDF (narrative extraction)
    try:
        doc = pymupdf.open(pdf_path)
        repeating_headers, repeating_footers = detect_repeating_header_footer(doc)
        if len(layouts) != len(doc):
            layouts = ["single"] * len(doc)

        for i, page in enumerate(doc):
            caps, rm = extract_figure_captions(page, i+1)
            all_captions.extend(caps)

            raw = (
                extract_double_column_page(page, repeating_headers, repeating_footers)
                if layouts[i] == "double"
                else extract_single_column_page(page, repeating_headers, repeating_footers)
            )
            for r in rm:
                raw = raw.replace(r, "")

            body_pages.append(raw)
            blocks_all.extend(extract_blocks_with_fonts(page))

        full_body = "\n\n".join(body_pages)
    except:
        full_body = extract_ocr(pdf_path)

    # Reconstruct paragraphs
    full_body = reconstruct_paragraphs(full_body)

    # Heading-based sectioning
    headings = detect_font_headings(blocks_all)
    headings = refine_headings(headings)
    headings = merge_multiline_headings(headings)
    sections = split_by_headings(blocks_all, headings)

    # Extract Title + Abstract
    title, abstract, sections = extract_title_and_abstract(full_body, sections)

    # Assemble final layout
    final_text = assemble_layout(title, abstract, sections, all_tables, all_captions)
    return final_text


# ============================================================
# 11. TITLE + ABSTRACT extraction
# ============================================================
def extract_title_and_abstract(full_body, sections):
    lines = [l.strip() for l in full_body.split("\n") if l.strip()]
    title = lines[0] if lines else ""

    abs_start = None
    for i, l in enumerate(lines):
        if l.lower().startswith("abstract"):
            abs_start = i
            break

    if abs_start is None:
        return title, "", sections

    abstract_lines = []
    for l in lines[abs_start+1:]:
        if is_heading_like(l):
            break
        abstract_lines.append(l)

    abstract = " ".join(abstract_lines).strip()

    # Remove abstract text from sections if duplicated
    return title, abstract, sections


# ============================================================
# 12. SAVE OUTPUT
# ============================================================
def save_text(pdf_path, text):
    fname = os.path.basename(pdf_path).replace(".pdf", ".txt")
    out = os.path.join(TEXT_FOLDER, fname)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out


# ============================================================
# 13. PARALLEL WORKER
# ============================================================
def process_pdf(fname):
    pdf_path = os.path.join(PDF_FOLDER, fname)
    try:
        text = extract_text_from_pdf(pdf_path)
        out = save_text(pdf_path, text)
        return fname, True, f"Saved → {out}"
    except Exception as e:
        return fname, False, f"Error: {e!r}"


# ============================================================
# 14. MAIN PARALLEL CONTROLLER
# ============================================================
def main():
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    not_to_process = [f for f in os.listdir(TEXT_FOLDER) if f.lower().endswith(".txt")]
    pdf_to_process = [f for f in pdfs if not f.replace(".pdf", ".txt") in not_to_process]

    if not pdfs:
        print("No PDFs found.")
        return

    maxw = int(os.getenv("MAX_WORKERS", "4"))
    print(f"Processing {len(pdfs)} PDFs with {maxw} workers...")

    with ProcessPoolExecutor(max_workers=maxw) as ex:
        fut = {ex.submit(process_pdf, f): f for f in pdf_to_process}
        for f in as_completed(fut):
            name = fut[f]
            try:
                fname, ok, msg = f.result()
                prefix = "✔" if ok else "⚠️"
                print(f"{prefix} {fname}: {msg}")
            except Exception as exc:
                print(f"💥 {name}: {exc!r}")

    print("PDF extraction complete.")


if __name__ == "__main__":
    main()
