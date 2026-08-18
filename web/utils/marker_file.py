"""Safe parsing for uploaded marker-gene tables."""

from pathlib import Path
from typing import Any
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException


MAX_ROWS = 100_000
MAX_COLUMNS = 1_000
MAX_XLSX_ARCHIVE_ENTRIES = 10_000
MAX_XLSX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


class MarkerFileError(ValueError):
    """Raised when an uploaded marker table is unsafe or unreadable."""


def get_upload_size(upload: Any) -> int | None:
    """Return the upload stream size without changing its read position."""
    stream = upload.stream
    try:
        position = stream.tell()
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(position)
    except (AttributeError, OSError):
        return None
    return size if size >= 0 else None


def _validate_xlsx_archive(upload: Any) -> None:
    """Reject malformed or disproportionately large XLSX archives before parsing."""
    stream = upload.stream
    stream.seek(0)
    try:
        with ZipFile(stream) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_XLSX_ARCHIVE_ENTRIES:
                raise MarkerFileError("XLSX file contains too many archive entries.")
            uncompressed_size = sum(entry.file_size for entry in entries)
            if uncompressed_size > MAX_XLSX_UNCOMPRESSED_BYTES:
                raise MarkerFileError(
                    "XLSX file expands beyond the 64 MB processing limit."
                )
    except BadZipFile as exc:
        raise MarkerFileError("File is not a valid XLSX workbook.") from exc
    finally:
        stream.seek(0)


def read_marker_dataframe(upload: Any) -> pd.DataFrame:
    """Parse a supported upload under bounded resource and error contracts."""
    filename = upload.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in {".csv", ".tsv", ".xlsx"}:
        raise MarkerFileError(
            "Unsupported file format. Please upload CSV, TSV, or XLSX files."
        )

    if extension == ".xlsx":
        _validate_xlsx_archive(upload)

    try:
        if extension == ".csv":
            dataframe = pd.read_csv(upload)
        elif extension == ".tsv":
            dataframe = pd.read_csv(upload, sep="\t")
        else:
            dataframe = pd.read_excel(upload)
    except pd.errors.EmptyDataError as exc:
        raise MarkerFileError("File is empty or contains no parseable data.") from exc
    except pd.errors.ParserError as exc:
        raise MarkerFileError("File contains malformed tabular data.") from exc
    except UnicodeDecodeError as exc:
        raise MarkerFileError(
            "File encoding is not supported. Please save the file as UTF-8."
        ) from exc
    except (
        BadZipFile,
        InvalidFileException,
        KeyError,
        OSError,
        OverflowError,
        ParseError,
        TypeError,
        ValueError,
    ) as exc:
        raise MarkerFileError("File could not be read as the selected format.") from exc

    if dataframe.empty:
        raise MarkerFileError("File is empty.")
    if dataframe.shape[0] > MAX_ROWS or dataframe.shape[1] > MAX_COLUMNS:
        raise MarkerFileError(
            "File dimensions are too large. Limit input to 100,000 rows and 1,000 columns."
        )
    return dataframe
