"""Извлечение текста договоров из PDF и DOCX."""

import re
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError
from pypdf import PdfReader
from pypdf.errors import PyPdfError

_WORD_END = re.compile(r"[^\W\d_][-\u00ad]$")


def extract_text_from_file(file_source: str | Path | BytesIO, file_name: str) -> str:
    """Извлечь текст из файла или байтового потока, сохранив его позицию.

    Формат определяется по расширению имени файла. Для пустого,
    нечитаемого или неподдерживаемого файла выбрасывается ValueError.
    """
    extension = Path(file_name).suffix.lower()
    if extension not in {".pdf", ".docx"}:
        raise ValueError("Формат файла не поддерживается. Загрузите PDF или DOCX.")

    try:
        content = (
            file_source.getvalue()
            if isinstance(file_source, BytesIO)
            else Path(file_source).read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise ValueError("Не удалось прочитать файл договора.") from exc

    if not content:
        raise ValueError("Файл договора пуст.")

    stream = BytesIO(content)
    try:
        if extension == ".pdf":
            text = "\n\n".join(_extract_pdf(stream))
        else:
            text = "\n\n".join(_extract_docx(stream))
    except (PyPdfError, PackageNotFoundError, BadZipFile, XMLSyntaxError, KeyError) as exc:
        raise ValueError(
            f"Не удалось прочитать {extension[1:].upper()}. "
            "Проверьте, что файл открывается и не защищён паролем."
        ) from exc

    if not text:
        message = "Не удалось извлечь текст из договора."
        if extension == ".pdf":
            message += " Если это скан, выполните распознавание текста и загрузите PDF снова."
        raise ValueError(message)
    return text


def _extract_pdf(stream: BytesIO) -> Iterator[str]:
    """Собрать абзацы PDF, используя пустые строки и границы страниц."""
    for page in PdfReader(stream).pages:
        page_text = page.extract_text(extraction_mode="layout")
        if not page_text.strip():
            page_text = page.extract_text() or ""

        lines: list[str] = []
        for raw_line in page_text.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                if lines:
                    yield " ".join(lines)
                    lines.clear()
                continue

            if lines and _WORD_END.search(lines[-1]) and line[0].islower():
                lines[-1] = lines[-1][:-1] + line
            else:
                lines.append(line)

        if lines:
            yield " ".join(lines)


def _extract_docx(stream: BytesIO) -> Iterator[str]:
    """Извлечь абзацы и строки таблиц в порядке элементов тела DOCX."""
    document = Document(stream)
    for element in document.element.body.iterchildren():
        if isinstance(element, CT_P):
            text = Paragraph(element, document).text.strip()
            if text:
                yield text
        elif isinstance(element, CT_Tbl):
            rows: list[str] = []
            for row in Table(element, document).rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                yield "\n".join(rows)
