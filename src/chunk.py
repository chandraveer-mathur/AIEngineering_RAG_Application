import re
import unicodedata

# --------------------------------------------------
# 1. Text cleanup
# --------------------------------------------------

def clean_text(text):
    """
    Normalize extracted text and remove known artifacts.
    """
    text = unicodedata.normalize("NFKC", text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove standalone numeric markers
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # Remove numeric markers at the beginning of a line
    text = re.sub(r"(?m)^\s*\d+\s+", "", text)

    # Join lines within a paragraph
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize spaces and blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# 2. Token counting
# --------------------------------------------------

def count_tokens(text, tokenizer):
    """
    Count the number of tokens in a piece of text.
    """
    return len(tokenizer.encode(text))


# --------------------------------------------------
# 3. Page extraction
# --------------------------------------------------

def extract_lines(page):
    """
    Extract text lines with font size and vertical position.
    """
    data = page.get_text("dict")
    lines = []

    for block in data["blocks"]:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            spans = line["spans"]

            text = clean_text(
                "".join(span["text"] for span in spans)
            )

            if not text:
                continue

            font_size = max(
                span["size"]
                for span in spans
            )

            lines.append({
                "text": text,
                "font_size": font_size,
                "y": line["bbox"][1],
            })

    # Read lines from top to bottom
    lines.sort(key=lambda line: line["y"])

    return lines


# --------------------------------------------------
# 4. Heading classification
# --------------------------------------------------

def classify_heading(text):
    """
    Classify a heading as a chapter or section.
    """
    if text.startswith("Chapter "):
        return "chapter"

    return "section"


# --------------------------------------------------
# 5. Extract page elements
# --------------------------------------------------

def extract_elements(page):
    """
    Separate headings from body text.
    """
    elements = []

    for line in extract_lines(page):
        text = line["text"]

        # Ignore small numeric markers
        if line["font_size"] < 12 and text.isdigit():
            continue

        # Large font = heading
        if line["font_size"] > 20:
            elements.append({
                "type": "heading",
                "heading_type": classify_heading(text),
                "text": text,
                "y": line["y"],
            })

        # Everything else = body text
        else:
            elements.append({
                "type": "text",
                "text": text,
                "y": line["y"],
            })

    return elements


# --------------------------------------------------
# 6. Convert lines into paragraphs
# --------------------------------------------------

def elements_to_paragraphs(elements, line_gap=35):
    """
    Combine body lines into paragraphs.

    Headings remain separate elements.
    A large vertical gap between body lines
    indicates a new paragraph.
    """
    paragraphs = []

    current_lines = []
    previous_y = None

    for element in elements:

        # ------------------------------------------
        # Heading
        # ------------------------------------------

        if element["type"] == "heading":

            # Save paragraph before heading
            if current_lines:
                paragraphs.append({
                    "type": "paragraph",
                    "text": " ".join(current_lines),
                })

                current_lines = []

            # Store heading separately
            paragraphs.append({
                "type": "heading",
                "heading_type": element["heading_type"],
                "text": element["text"],
            })

            previous_y = None

            continue

        # ------------------------------------------
        # Body text
        # ------------------------------------------

        current_y = element["y"]

        if previous_y is not None:
            gap = current_y - previous_y

            # Large vertical gap = new paragraph
            if gap > line_gap:

                paragraphs.append({
                    "type": "paragraph",
                    "text": " ".join(current_lines),
                })

                current_lines = []

        current_lines.append(element["text"])

        previous_y = current_y

    # Save final paragraph
    if current_lines:
        paragraphs.append({
            "type": "paragraph",
            "text": " ".join(current_lines),
        })

    return paragraphs


# --------------------------------------------------
# 7. Create document chunks
# --------------------------------------------------

def create_document_chunks(
    doc,
    tokenizer,
    target_tokens=400,
):
    """
    Process the PDF in page order.
    Preserve chapter, section, and page information.
    A heading starts a new chunk.
    Long sections can produce multiple chunks
    based on the target token count.
    """

    chunks = []

    current_chapter = None
    current_section = None

    current_paragraphs = []
    current_token_count = 0

    current_page_start = None
    current_page_end = None

    # ----------------------------------------------
    # Save current chunk
    # ----------------------------------------------

    def save_current_chunk():
        """
        Save the current chunk if it contains text.
        """

        nonlocal current_paragraphs
        nonlocal current_token_count
        nonlocal current_page_start
        nonlocal current_page_end

        if not current_paragraphs:
            return

        chunks.append({
            "text": "\n\n".join(current_paragraphs),
            "token_count": current_token_count,
            "chapter": current_chapter,
            "section": current_section,
            "page_start": current_page_start,
            "page_end": current_page_end,
        })

        # Reset chunk state
        current_paragraphs = []
        current_token_count = 0
        current_page_start = None
        current_page_end = None

    # ----------------------------------------------
    # Process document page by page
    # ----------------------------------------------

    for page_number, page in enumerate(doc, start=1):

        elements = extract_elements(page)

        paragraphs = elements_to_paragraphs(elements)

        for paragraph in paragraphs:

            # --------------------------------------
            # Heading
            # --------------------------------------

            if paragraph["type"] == "heading":

                # Heading starts a new chunk
                save_current_chunk()

                if paragraph["heading_type"] == "chapter":

                    current_chapter = paragraph["text"]
                    current_section = None

                else:

                    current_section = paragraph["text"]

                continue

            # --------------------------------------
            # Paragraph
            # --------------------------------------

            text = paragraph["text"]

            token_count = count_tokens(
                text,
                tokenizer,
            )

            # First paragraph in a new chunk
            if not current_paragraphs:
                current_page_start = page_number

            # --------------------------------------
            # Does paragraph fit?
            # --------------------------------------

            if (
                current_token_count > 0
                and current_token_count + token_count > target_tokens
            ):

                # Save existing chunk
                save_current_chunk()

                # New chunk starts on this page
                current_page_start = page_number

            # Add paragraph
            current_paragraphs.append(text)

            current_token_count += token_count

            current_page_end = page_number

    # ----------------------------------------------
    # Save final chunk
    # ----------------------------------------------

    save_current_chunk()

    return chunks