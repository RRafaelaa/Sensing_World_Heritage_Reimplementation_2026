import os
import pdfplumber
import pymupdf
import pytesseract
from pdf2image import convert_from_path


PDF_FOLDER = "data/pdfs"
TEXT_FOLDER = "data/texts"
os.makedirs(TEXT_FOLDER, exist_ok=True)

def extract_pdfplumber(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
        return text.strip()
    except Exception:
        return ""


def extract_pymupdf(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n\n"
        return text.strip()
    except Exception:
        return ""


def extract_ocr(pdf_path):
    try:
        pages = convert_from_path(pdf_path)
        text = ""
        for img in pages:
            text += pytesseract.image_to_string(img) + "\n\n"
        return text.strip()
    except Exception:
        return ""


def save_text(pdf_path, text):
    basename = os.path.basename(pdf_path)
    txt_name = basename.replace(".pdf", ".txt")
    out_path = os.path.join(TEXT_FOLDER, txt_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    return out_path


def extract_text(pdf_path):
    text = extract_pdfplumber(pdf_path)
    if text:
        return save_text(pdf_path, text)

    text = extract_pymupdf(pdf_path)
    if text:
        return save_text(pdf_path, text)

    text = extract_ocr(pdf_path)
    return save_text(pdf_path, text)


def main():
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    not_to_process = [f for f in os.listdir(TEXT_FOLDER) if f.lower().endswith(".txt")]
    pdf_to_process = [f for f in pdfs if not f.replace(".pdf", ".txt") in not_to_process]

    if not pdfs:
        print("No PDFs found.")
        return

    for pdf_file in pdf_to_process:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"Extracting: {pdf_file}")
        txt_path = extract_text(pdf_path)
        print(f"Saved → {txt_path}")

    print("PDF → text extraction completed.")


if __name__ == "__main__":
    main()
