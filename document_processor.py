import fitz
from PIL import Image
import pytesseract
import io
import os
import re


def _clean_text(text: str) -> str:
    """Normalize extracted text for downstream RAG.

    - Normalize unicode fractions to decimal strings so numbers remain faithful.
    - Collapse multiple whitespace characters into a single space/newline.
    - Strip trailing spaces on each line.
    """
    if not text:
        return ""

    # Normalize common unicode fractions that appear in IMF PDFs
    text = text.replace("½", "0.5").replace("¼", "0.25").replace("¾", "0.75")

    # Normalize weird unicode minus or spaces around numbers if any crop up
    text = text.replace("−", "-")

    # Collapse consecutive spaces but preserve newlines
    # First normalise spaces inside each line, then rebuild the text.
    cleaned_lines = []
    for line in text.split("\n"):
        # Replace multiple whitespace chars with a single space
        line = re.sub(r"\s+", " ", line).strip()
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


class DocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    # -----------------------------
    # Text extraction and chunking
    # -----------------------------
    @staticmethod
    def _chunk_text(text: str, min_chars: int = 300, max_chars: int = 500):
        """
        Split long text into reasonably sized chunks (300–500 characters).
        Uses paragraph boundaries when possible, but enforces length limits.
        """
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            if not current:
                current = para
            elif len(current) + 1 + len(para) <= max_chars:
                current = current + "\n" + para
            else:
                # Current chunk would be too long if we add this paragraph
                if len(current) >= min_chars:
                    chunks.append(current)
                    current = para
                else:
                    # Current is short; force-merge
                    current = current + " " + para
                    if len(current) >= min_chars:
                        chunks.append(current)
                        current = ""

        if current:
            # Last chunk; if it's very short, merge with previous when possible
            if chunks and len(current) < min_chars:
                chunks[-1] = chunks[-1] + "\n" + current
            else:
                chunks.append(current)

        return chunks

    def extract_text_chunks(self):
        """
        Extract text and split into 300–500 character chunks.
        Adds metadata: page number, type, section, and source.
        """
        chunks = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()

            if text and text.strip():
                # Clean raw page text before chunking to normalise unicode and spaces
                text = _clean_text(text)
                raw_chunks = self._chunk_text(text)
                for idx, chunk_text in enumerate(raw_chunks):
                    section_label = f"page_{page_num + 1}_section_{idx + 1}"
                    chunks.append(
                        {
                            "type": "text",
                            "content": chunk_text,
                            "page": page_num + 1,
                            "section": section_label,
                            "source": f"Page {page_num + 1} - {section_label}",
                        }
                    )

        return chunks

    # -----------------------------
    # Table extraction
    # -----------------------------
    def extract_tables(self):
        """
        Keep tables as separate chunks with their own metadata.
        """
        tables = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    lines = block["lines"]
                    if len(lines) > 2:
                        table_text = ""
                        for line in lines:
                            for span in line["spans"]:
                                table_text += span["text"] + " "
                            table_text += "\n"

                        if table_text.strip():
                            section_label = "table"
                            tables.append(
                                {
                                    "type": "table",
                                    "content": _clean_text(table_text),
                                    "page": page_num + 1,
                                    "section": section_label,
                                    "source": f"Table on Page {page_num + 1}",
                                }
                            )

        return tables

    # -----------------------------
    # Image extraction + OCR
    # -----------------------------
    def extract_images_with_ocr(self, output_folder=None):
        if output_folder is None:
            try:
                import config

                output_folder = config.IMAGES_DIR
            except Exception:
                output_folder = "extracted_images"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images_data = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]

                image_filename = f"{output_folder}/page{page_num + 1}_img{img_index + 1}.png"
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)

                try:
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(img_pil)

                    if ocr_text.strip():
                        images_data.append(
                            {
                                "type": "image",
                                "content": _clean_text(ocr_text),
                                "page": page_num + 1,
                                "section": "image",
                                "image_path": image_filename,
                                "source": f"Image on Page {page_num + 1}",
                            }
                        )
                except Exception as e:
                    print(f"OCR failed on page {page_num + 1}: {e}")

        return images_data

    # -----------------------------
    # End-to-end processing
    # -----------------------------
    def process_document(self):
        print(f"Processing document: {self.pdf_path}")

        text_chunks = self.extract_text_chunks()
        print(f"Extracted {len(text_chunks)} text chunks")

        tables = self.extract_tables()
        print(f"Extracted {len(tables)} tables")

        images = self.extract_images_with_ocr()
        print(f"Extracted {len(images)} images with OCR")

        all_chunks = text_chunks + tables + images
        print(f" Total chunks: {len(all_chunks)}")

        return all_chunks

    def close(self):
        self.doc.close()


if __name__ == "__main__":
    processor = DocumentProcessor("qatar_test_doc.pdf")
    chunks = processor.process_document()
    if chunks:
        print(f"\nSample chunk: {chunks[0]}")
    processor.close()
