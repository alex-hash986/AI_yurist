"""Создание юридического заключения Word в памяти."""

from collections.abc import Iterable, Sequence
from io import BytesIO

from docx import Document
from docx.document import Document as WordDocument
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.text.paragraph import Paragraph

from core.analyzer import ComparisonStatus, ContractAnalysisReport, RiskLevel

_TITLE = "Заключение по результатам AI-анализа договора"
_DISCLAIMER = (
    "Внимание: отчёт сформирован AI-ассистентом в информационных целях "
    "и не заменяет консультацию квалифицированного юриста."
)
_RISK_LABELS: dict[RiskLevel, tuple[str, str]] = {
    "high": ("Высокий", "A4262C"),
    "medium": ("Средний", "8A5100"),
    "low": ("Низкий", "216242"),
}
_STATUS_LABELS: dict[ComparisonStatus, str] = {
    "ok": "Соответствует",
    "deviation": "Отклонение",
    "critical": "Критично",
}


def generate_docx_report(report: ContractAnalysisReport, file_name: str) -> bytes:
    """Сформировать заключение и вернуть байты DOCX для скачивания."""
    doc = Document()
    doc.core_properties.title = _TITLE
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2)
    section.top_margin = section.bottom_margin = Cm(2)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (("Title", 20), ("Heading 1", 15), ("Heading 2", 12)):
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        for border in style.element.xpath("./w:pPr/w:pBdr"):
            border.getparent().remove(border)

    doc.add_paragraph(_TITLE, style="Title")
    _labelled(doc, "Файл", file_name)
    _labelled(doc, "Тип договора", report.contract_type)
    label, color = _RISK_LABELS[report.overall_risk_score]
    _labelled(doc, "Общий уровень риска", label, color=color)
    notice = doc.add_paragraph()
    notice.add_run(_DISCLAIMER).bold = True
    shade = OxmlElement("w:shd")
    shade.set(qn("w:fill"), "FFF2CC")
    notice._p.get_or_add_pPr().insert_element_before(shade, "w:spacing", "w:ind")

    doc.add_heading("1 Резюме проверки", level=1)
    doc.add_paragraph(report.executive_summary)

    doc.add_heading("2 Ключевые условия договора", level=1)
    _add_table(
        doc,
        ("Условие", "Содержание", "Пункт"),
        (4, 10, 3),
        ((item.title, item.value, item.clause_ref) for item in report.key_conditions),
    )

    doc.add_heading("3 Выявленные риски и рекомендации", level=1)
    if not report.risks:
        doc.add_paragraph("В отчёте нет рисков с подтверждёнными цитатами.")
    for risk in report.risks:
        doc.add_heading(risk.title, level=2)
        label, color = _RISK_LABELS[risk.level]
        _labelled(doc, "Уровень риска", label, color=color)
        _labelled(doc, "Цитата из договора", risk.exact_quote)
        _labelled(doc, "Правовое обоснование", risk.explanation)
        if risk.precedent_ref is not None:
            _labelled(doc, "Источник из базы знаний", risk.precedent_ref)
        _labelled(doc, "Редакция для протокола разногласий", risk.recommendation)

    doc.add_heading("4 Сравнение с типовым договором", level=1)
    _add_table(
        doc,
        ("Параметр", "Договор", "Эталон", "Статус", "Комментарий"),
        (3, 3.6, 3.6, 2.7, 4.1),
        (
            (
                item.aspect,
                item.contract_value,
                item.template_value,
                _STATUS_LABELS[item.status],
                item.comment,
            )
            for item in report.template_comparison
        ),
    )

    doc.add_heading("5 Нестандартные пункты и источники", level=1)
    doc.add_heading("Нестандартные пункты", level=2)
    if not report.non_standard_clauses:
        doc.add_paragraph("Нестандартные условия в отчёте не указаны.")
    for clause in report.non_standard_clauses:
        doc.add_paragraph(clause, style="List Bullet")

    for heading, sources in (
        ("Обзоры юридических рисков", report.retrieved_precedents),
        ("Эталонные шаблоны", report.retrieved_templates),
    ):
        doc.add_heading(heading, level=2)
        if not sources:
            doc.add_paragraph("Фрагменты из этой коллекции отсутствуют.")
        for source in sources:
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.keep_with_next = True
            paragraph.add_run(source.get("source", "")).bold = True
            doc.add_paragraph(source.get("text", ""))

    stream = BytesIO()
    doc.save(stream)
    return stream.getvalue()


def _labelled(
    doc: WordDocument, label: str, text: str, *, color: str | None = None
) -> Paragraph:
    """Добавить абзац с полужирной меткой и необязательным цветом значения."""
    paragraph = doc.add_paragraph()
    paragraph.add_run(f"{label}: ").bold = True
    run = paragraph.add_run(text)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
        run.bold = True
    return paragraph


def _add_table(
    doc: WordDocument, headers: Sequence[str], widths: Sequence[float],
    rows: Iterable[Sequence[str]],
) -> None:
    """Добавить таблицу с фиксированными ширинами и повторяемой шапкой."""
    values = list(rows)
    if not values:
        doc.add_paragraph("Нет данных для таблицы.")
        return
    table = doc.add_table(rows=0, cols=len(headers), style="Table Grid")
    table.autofit = False
    for column, width in zip(table.columns, widths, strict=True):
        column.width = Cm(width)

    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        for key, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            border.set(qn(f"w:{key}"), value)
        borders.append(border)
    table._tbl.tblPr.insert_element_before(borders, "w:shd", "w:tblLayout")

    for index, row_values in enumerate([headers, *values]):
        row = table.add_row()
        if index == 0:
            row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for cell, text in zip(row.cells, row_values, strict=True):
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(4)
            run = paragraph.add_run(text)
            run.font.size = Pt(10)
            if index == 0:
                run.bold = True
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "E9EDF2")
                cell._tc.get_or_add_tcPr().append(shade)
