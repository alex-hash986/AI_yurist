"""Собрать техническое задание проекта в формате Word."""

from collections.abc import Sequence
from pathlib import Path

from docx import Document
from docx.document import Document as WordDocument
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "Техническое_задание_AI_Юрист.docx"
TITLE = (
    "Техническое задание на разработку системы "
    "«AI-ассистент для первичного юридического анализа договоров»"
)
DISCLAIMER = (
    "Учебный/демонстрационный проект. Используются публичные и синтетические шаблоны "
    "договоров. Система не заменяет консультацию квалифицированного юриста."
)


def _configure(document: WordDocument) -> None:
    """Настроить страницу, типографику и нумерацию страниц."""
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2)
    section.top_margin = section.bottom_margin = Cm(1.9)
    section.footer_distance = Cm(0.9)

    normal = document.styles["Normal"]
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for name, size in (("Normal", 11), ("Title", 24), ("Heading 1", 15),
                       ("List Bullet", 11), ("List Number", 11)):
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        for border in style.element.xpath("./w:pPr/w:pBdr"):
            border.getparent().remove(border)
        for font in style.element.xpath("./w:rPr/w:rFonts"):
            for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
                font.attrib.pop(qn(f"w:{attribute}"), None)
    heading = document.styles["Heading 1"].paragraph_format
    heading.space_before = Pt(12)
    heading.space_after = Pt(6)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.paragraph_format.space_after = Pt(0)
    run = footer.add_run("Версия 1.0  ·  ")
    run.font.size = Pt(9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    document.core_properties.title = TITLE
    document.core_properties.version = "1.0"


def _disclaimer(document: WordDocument) -> None:
    """Добавить выделенный абзац с рамкой и фоном."""
    paragraph = document.add_paragraph(DISCLAIMER)
    paragraph.paragraph_format.left_indent = Cm(0.25)
    paragraph.paragraph_format.right_indent = Cm(0.25)
    paragraph.paragraph_format.space_after = Pt(9)
    paragraph.runs[0].font.size = Pt(10)
    properties = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{edge}")
        for name, value in (("val", "single"), ("sz", "8"), ("space", "6"), ("color", "B6C2D0")):
            border.set(qn(f"w:{name}"), value)
        borders.append(border)
    shade = OxmlElement("w:shd")
    shade.set(qn("w:fill"), "F1F5F9")
    properties.insert_element_before(borders, "w:spacing", "w:ind")
    properties.insert_element_before(shade, "w:spacing", "w:ind")


def _table(
    document: WordDocument,
    widths: Sequence[float],
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    """Добавить таблицу с заданными ширинами и повторяемой шапкой."""
    table = document.add_table(rows=0, cols=len(headers), style="Table Grid")
    table.autofit = False
    for column, width in zip(table.columns, widths, strict=True):
        column.width = Cm(width)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        for name, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            border.set(qn(f"w:{name}"), value)
        borders.append(border)
    table._tbl.tblPr.insert_element_before(borders, "w:shd", "w:tblLayout")

    for index, values in enumerate((headers, *rows)):
        row = table.add_row()
        properties = row._tr.get_or_add_trPr()
        properties.append(OxmlElement("w:cantSplit"))
        if index == 0:
            properties.append(OxmlElement("w:tblHeader"))
        for cell, value in zip(row.cells, values, strict=True):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(4)
            paragraph.paragraph_format.line_spacing = 1.06
            run = paragraph.add_run(value)
            run.font.size = Pt(10)
            run.bold = index == 0
            if index == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "E8EEF5")
                cell._tc.get_or_add_tcPr().append(shade)


def _list(document: WordDocument, items: Sequence[str], *, numbered: bool = False) -> None:
    """Добавить маркированный или нумерованный список Word."""
    for text in items:
        document.add_paragraph(text, style="List Number" if numbered else "List Bullet")


def generate_tz_docx(output: Path = OUTPUT) -> Path:
    """Сформировать шесть разделов ТЗ и сохранить документ."""
    document = Document()
    _configure(document)
    document.add_paragraph(TITLE, style="Title")
    document.add_paragraph("Версия 1.0").runs[0].bold = True
    _disclaimer(document)

    document.add_heading("1 Назначение и цели проекта", level=1)
    document.add_paragraph(
        "Система предназначена для первичного скрининга B2B-договоров перед согласованием "
        "и подготовки черновика заключения для проверки человеком."
    )
    _list(document, (
        "Автоматизировать первичный скрининг договоров.",
        "Выявлять скрытые финансовые и юридические риски до подписания.",
        "Сокращать рутинную нагрузку на юристов при поиске и сопоставлении условий.",
        "Стандартизировать проверку по корпоративным эталонам.",
    ))
    document.add_heading("2 Целевая аудитория", level=1)
    _table(document, (4.0, 6.1, 6.9), ("Сегмент", "Текущие проблемы", "Как помогает система"), (
        ("Штатные корпоративные юристы",
         "Повторяющиеся проверки и ручная сверка с внутренними стандартами.",
         "Собирает условия и отклонения, показывает цитаты и готовит черновик замечаний."),
        ("Собственники малого бизнеса и ИП без штатного юриста",
         "Сложно оценить финансовые последствия пункта и решить, когда привлечь юриста.",
         "Выделяет приоритетные вопросы и даёт материал для предметной консультации."),
        ("Менеджеры отделов продаж и закупок",
         "Согласование замедляют разрозненные замечания и неожиданные правки контрагента.",
         "Показывает отличия от шаблона и даёт редакции для обсуждения с юристом."),
    ))

    heading = document.add_heading("3 Функциональные требования", level=1)
    heading.paragraph_format.page_break_before = True
    _list(document, (
        "Принимать PDF с текстовым слоем и DOCX. Извлекать абзацы и таблицы реквизитов "
        "и спецификаций DOCX в порядке следования. Для нечитаемых файлов показывать понятную ошибку.",
        "Анализировать договор с позиции выбранной стороны. Доступны роли "
        "«Исполнитель / Подрядчик» и «Заказчик».",
        "Извлекать ключевые и нестандартные условия. Классифицировать риски как высокие, "
        "средние или низкие, подтверждать их дословными цитатами и предлагать формулировки "
        "для протокола разногласий.",
        "Выполнять векторный поиск по обзорам практики и рисков и сравнивать условия "
        "договора с эталонными шаблонами из ChromaDB.",
        "Показывать документ и цветовую HTML-подсветку цитат слева. Справа выводить "
        "аналитику, фильтр рисков, сравнение с эталоном и источники RAG с релевантностью.",
        "Экспортировать резюме, ключевые условия, риски, рекомендации, сравнение "
        "и использованные материалы в структурированное заключение Microsoft Word (.docx).",
    ))
    document.add_heading("4 Технические требования и архитектура", level=1)
    _table(document, (4.2, 12.8), ("Компонент", "Реализация"), (
        ("Интерфейс и ввод", "Python и Streamlit. pypdf извлекает текст PDF, python-docx читает и формирует DOCX."),
        ("Поиск по базе знаний", "LangChain нарезает тексты и получает эмбеддинги через OpenAI. ChromaDB хранит локальные коллекции шаблонов и обзоров рисков."),
        ("Анализ", "OpenAI API со Structured Outputs формирует результат по схеме Pydantic. Приложение сверяет цитаты с исходным текстом."),
        ("Конфигурация и хранение", "Ключ задаётся через .env или окружение. В Git хранится только шаблон настроек. Рабочий индекс ChromaDB хранится локально вне Git."),
    ))
    document.add_paragraph(
        "Текущая реализация передаёт текст договора, запросы и выдержки базы знаний в OpenAI. "
        "Для конфиденциальных договоров нужен согласованный режим обработки данных."
    )

    heading = document.add_heading("5 Ограничения, допущения и риски проекта", level=1)
    heading.paragraph_format.page_break_before = True
    _table(document, (4.2, 5.6, 7.2), ("Риск или ограничение", "Последствие", "Мера"), (
        ("Бюджет и лимиты OpenAI API", "Расходы растут с числом запросов. Лимит может прервать анализ.",
         "Контролировать квоту и расход. Показывать ошибку API и сохранять готовый отчёт в сессии для повторного просмотра."),
        ("PDF без текстового слоя", "Скан без распознанного текста нельзя проанализировать.",
         "Показывать сообщение об ошибке. Загружать PDF после внешнего OCR или использовать DOCX."),
        ("Ошибки и галлюцинации модели", "Возможны неверные выводы и ошибочные ссылки.",
         "Использовать RAG, сверять дословность цитат и проверять ссылки на источники. Юрист проверяет выводы."),
        ("Устаревание базы знаний", "Нормы, разъяснения и эталоны могут измениться.",
         "Пересматривать материалы по графику и переиндексировать коллекции после обновления."),
        ("Справочный статус заключения", "Автоматический вывод требует юридической оценки перед применением.",
         "Показывать дисклеймер в интерфейсе и отчёте. Передавать спорные условия квалифицированному юристу."),
    ))
    document.add_heading("6 Критерии приёмки и передаваемые материалы", level=1)
    _list(document, (
        "Приложение запускается через streamlit run app.py. Доступны загрузка файла и оба тестовых договора.",
        "PDF с текстовым слоем и DOCX читаются. Таблицы DOCX выводятся в исходном порядке, скан без текста вызывает понятную ошибку.",
        "Обе роли доступны. Отчёт содержит дословные подсвеченные цитаты, уровни риска, рекомендации, сравнение и найденные источники.",
        "Скачанный DOCX открывается в Word, содержит структурированные разделы, таблицы и дисклеймер.",
    ), numbered=True)
    document.add_paragraph(
        "Передаваемые материалы — исходный код и зависимости, README, шаблон конфигурации "
        "без ключа, материалы базы знаний, два тестовых договора, ТЗ и бизнес-кейс."
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    return output


if __name__ == "__main__":
    print(generate_tz_docx())
