from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_extracted_text(text: str) -> str:
    """Clean common spacing issues from extracted PDF text.

    PDF extraction often returns one visual line at a time. Joining those lines
    into paragraphs makes the preview and retrieved chunks easier to read.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return ""

    paragraphs = [lines[0]]
    current_paragraph = []

    for line in lines[1:]:
        if len(line) <= 60 and not line.endswith((".", ",", ";", ":")):
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            paragraphs.append(line)
        else:
            current_paragraph.append(line)

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    return "\n\n".join(paragraphs)


def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split document text into smaller overlapping chunks for retrieval."""
    cleaned_text = clean_extracted_text(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return text_splitter.split_text(cleaned_text)
