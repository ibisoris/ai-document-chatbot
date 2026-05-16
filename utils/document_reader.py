import csv
import json
from pathlib import Path

import pandas as pd

from utils.pdf_reader import PdfExtractionError, extract_text_from_pdf, get_pdf_page_count
from utils.text_chunker import clean_extracted_text


class DocumentReadError(Exception):
    """Raised when an uploaded document cannot be read."""


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".json", ".xlsx", ".xls"}


def get_document_kind(file_path: Path) -> str:
    """Classify the uploaded file as structured or unstructured."""
    if file_path.suffix.lower() in {".csv", ".json", ".xlsx", ".xls"}:
        return "structured"

    return "unstructured"


def read_text_file(file_path: Path) -> tuple[str, int | None]:
    """Read a plain text file as unstructured document text."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")
    except OSError as error:
        raise DocumentReadError("The text file could not be read.") from error

    return clean_extracted_text(text), None


def read_csv_file(file_path: Path) -> tuple[str, int | None]:
    """Convert a CSV file into searchable row-oriented text."""
    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
    except csv.Error as error:
        raise DocumentReadError("The CSV file could not be parsed.") from error
    except OSError as error:
        raise DocumentReadError("The CSV file could not be read.") from error

    if not rows:
        raise DocumentReadError("The CSV file does not contain readable rows.")

    columns = reader.fieldnames or []
    text_rows = [
        f"Structured CSV dataset with columns: {', '.join(columns)}.",
        f"Total rows: {len(rows)}.",
    ]

    for index, row in enumerate(rows, start=1):
        values = [
            f"{column}: {str(row.get(column, '')).strip()}"
            for column in columns
            if str(row.get(column, "")).strip()
        ]
        text_rows.append(f"Row {index}: " + "; ".join(values))

    return "\n\n".join(text_rows), len(rows)


def read_json_file(file_path: Path) -> tuple[str, int | None]:
    """Convert JSON objects or arrays into searchable structured text."""
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise DocumentReadError("The JSON file could not be parsed.") from error
    except OSError as error:
        raise DocumentReadError("The JSON file could not be read.") from error

    if isinstance(data, list):
        records = data
    else:
        records = [data]

    text_rows = [
        "Structured JSON document.",
        f"Total records: {len(records)}.",
    ]

    for index, record in enumerate(records, start=1):
        formatted_record = json.dumps(record, ensure_ascii=True)
        text_rows.append(f"Record {index}: {formatted_record}")

    return "\n\n".join(text_rows), len(records)


def read_excel_file(file_path: Path) -> tuple[str, int | None]:
    """Convert Excel worksheets into searchable sheet and row text."""
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)
    except Exception as error:
        raise DocumentReadError("The Excel file could not be parsed.") from error

    if not sheets:
        raise DocumentReadError("The Excel file does not contain readable sheets.")

    text_rows = ["Structured Excel workbook."]
    total_records = 0

    for sheet_name, dataframe in sheets.items():
        dataframe = dataframe.dropna(how="all")
        columns = [str(column) for column in dataframe.columns]
        total_records += len(dataframe)

        text_rows.append(
            f"Sheet '{sheet_name}' with columns: {', '.join(columns)}."
        )
        text_rows.append(f"Sheet '{sheet_name}' row count: {len(dataframe)}.")

        for index, row in dataframe.iterrows():
            values = [
                f"{column}: {row[column]}"
                for column in dataframe.columns
                if pd.notna(row[column])
            ]
            text_rows.append(
                f"Sheet '{sheet_name}', row {index + 1}: " + "; ".join(values)
            )

    return "\n\n".join(text_rows), total_records


def read_document(file_path: Path) -> dict:
    """Read supported structured and unstructured documents for RAG indexing."""
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentReadError(
            "Unsupported file type. Upload a PDF, TXT, CSV, JSON, or Excel file."
        )

    if extension == ".pdf":
        try:
            return {
                "text": extract_text_from_pdf(file_path),
                "page_count": get_pdf_page_count(file_path),
                "record_count": None,
                "document_kind": "unstructured",
            }
        except PdfExtractionError as error:
            raise DocumentReadError(
                "I could not read this PDF. Please try another PDF file."
            ) from error

    if extension == ".txt":
        text, record_count = read_text_file(file_path)
    elif extension == ".csv":
        text, record_count = read_csv_file(file_path)
    elif extension == ".json":
        text, record_count = read_json_file(file_path)
    else:
        text, record_count = read_excel_file(file_path)

    return {
        "text": text,
        "page_count": None,
        "record_count": record_count,
        "document_kind": get_document_kind(file_path),
    }
