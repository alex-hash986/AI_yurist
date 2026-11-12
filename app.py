"""Двухпанельное рабочее пространство для юридического анализа договоров."""

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from hashlib import sha256
from html import escape
from io import BytesIO
from pathlib import Path

import streamlit as st
from chromadb.errors import ChromaError

from config import SAMPLES_DIR
from core.analyzer import ContractAnalysisReport, ContractAnalyzer, RetrievedFragment, RiskItem
from core.parser import extract_text_from_file
from core.report_gen import generate_docx_report
from core.vector_store import LegalVectorStore

_UPLOAD = "Загрузить файл (PDF / DOCX)"
_SAMPLES = "Тестовый договор из базы"
_LEVELS = {"high": "Высокий", "medium": "Средний", "low": "Низкий"}
_PRIORITY = {level: rank for rank, level in enumerate(_LEVELS)}
_STATUSES = {
    "ok": "✅ Соответствует",
    "deviation": "⚠️ Отклонение",
    "critical": "🚨 Критическое расхождение",
}
_PARAGRAPH_BREAK = re.compile(r"(\n(?:[^\S\n]*\n)+)")
_SOURCE_URL = re.compile(r"""https://[^\s<>"']+""")
_CSS = """
<style>
.contract-paper {
    background: #f8fafc; color: #1e293b; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: clamp(1rem, 3vw, 1.75rem);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.05rem; line-height: 1.65; overflow-wrap: anywhere;
}
.contract-paper p { margin: 0 0 1em; white-space: pre-wrap; }
.contract-paper p:last-child { margin-bottom: 0; }
.risk-quote { margin: .75rem 0; white-space: pre-wrap; overflow-wrap: anywhere; }
mark.highlight-high, mark.highlight-medium, mark.highlight-low {
    color: #1e293b; border-radius: .3rem; padding: .12em .35em;
    box-decoration-break: clone; -webkit-box-decoration-break: clone;
}
.highlight-high { background: #fee2e2; border-left: 4px solid #dc2626; }
.highlight-medium { background: #fef3c7; border-left: 4px solid #d97706; }
.highlight-low { background: #e0f2fe; border-left: 4px solid #0284c7; }
.risk-badge {
    display: inline-block; vertical-align: baseline; white-space: normal;
    color: #1e293b; border-radius: .35rem; padding: .12rem .45rem;
    margin: 0 .35rem .15rem 0; font: 700 .73rem/1.4 system-ui, sans-serif;
}
</style>
"""


def _badge(level: str, title: str = "") -> str:
    """Сформировать бейдж с экранированным названием риска."""
    label = f"{_LEVELS[level].upper()} РИСК"
    if title:
        label += f": {title}"
    return f'<span class="risk-badge highlight-{level}">{escape(label)}</span>'


def render_contract_html(text: str, risks: Sequence[RiskItem] = ()) -> str:
    """Подсветить все совпадения цитат по смещениям в исходном тексте."""
    starts: dict[int, list[int]] = defaultdict(list)
    ends: dict[int, list[int]] = defaultdict(list)
    for index, risk in enumerate(risks):
        if not risk.exact_quote.strip():
            continue
        offset = text.find(risk.exact_quote)
        while offset != -1:
            starts[offset].append(index)
            ends[offset + len(risk.exact_quote)].append(index)
            offset = text.find(risk.exact_quote, offset + 1)

    # Разделители участвуют в проходе по смещениям, чтобы цитаты могли
    # пересекать границу абзаца, не нарушая структуру HTML.
    breaks = {match.start(): match.end() for match in _PARAGRAPH_BREAK.finditer(text)}
    boundaries = sorted({0, len(text), *starts, *ends, *breaks, *breaks.values()})
    active: Counter[int] = Counter()
    pending_badges: list[int] = []
    pieces = ['<div class="contract-paper"><p>']
    skip_until = 0
    for start, end in zip(boundaries, boundaries[1:]):
        for index in ends.get(start, ()):
            active[index] -= 1
            if not active[index]:
                del active[index]
        for index in starts.get(start, ()):
            active[index] += 1
            pending_badges.append(index)

        if start in breaks:
            skip_until = breaks[start]
            pieces.append("</p><p>")
        if start < skip_until:
            continue
        for index in pending_badges:
            risk = risks[index]
            pieces.append(_badge(risk.level, risk.title))
        pending_badges.clear()

        part = escape(text[start:end])
        if active:
            level = min((risks[index].level for index in active), key=_PRIORITY.__getitem__)
            part = f'<mark class="highlight-{level}">{part}</mark>'
        pieces.append(part)
    pieces.append("</p></div>")
    return "".join(pieces)


def _reset_report() -> None:
    """Удалить заключение и экспорт для предыдущего документа или стороны."""
    st.session_state.report = None
    st.session_state.docx_bytes = None


def _set_document(source: str, name: str, content: bytes | None) -> None:
    """Извлечь текст при изменении файла и сохранить результат в сессии."""
    identity = (source, name, sha256(content).digest()) if content is not None else None
    if "document_id" in st.session_state and st.session_state.document_id == identity:
        return
    _reset_report()
    st.session_state.document_id = identity
    st.session_state.file_name = name
    st.session_state.contract_text = ""
    st.session_state.document_error = None
    if content is not None:
        try:
            st.session_state.contract_text = extract_text_from_file(BytesIO(content), name)
        except ValueError as exc:
            st.session_state.document_error = str(exc)


def _sidebar() -> None:
    """Показать настройки документа, действия анализа и управление индексом."""
    with st.sidebar:
        st.header("⚖️ Юридический анализ")
        st.caption("Проверка договоров по эталонным условиям и обзорам юридических рисков.")
        source = st.radio("Источник документа", [_UPLOAD, _SAMPLES])
        if source == _UPLOAD:
            uploaded = st.file_uploader("Файл договора", type=["pdf", "docx"])
            _set_document(source, uploaded.name if uploaded else "", uploaded.getvalue() if uploaded else None)
        else:
            samples = sorted(
                path.name for path in SAMPLES_DIR.iterdir()
                if path.is_file() and path.suffix.lower() in {".docx", ".pdf"}
            ) if SAMPLES_DIR.is_dir() else []
            name = st.selectbox("Тестовый договор", samples, disabled=not samples, key="sample_file")
            try:
                _set_document(source, name or "", (SAMPLES_DIR / name).read_bytes() if name else None)
            except OSError:
                _set_document(source, "", None)
                st.error("Не удалось прочитать тестовый договор.")

        role = st.radio("Сторона анализа", ["Исполнитель / Подрядчик", "Заказчик"], key="role_perspective")
        if st.session_state.get("report_perspective") != role:
            _reset_report()
            st.session_state.report_perspective = role

        if st.button(
            "🔍 Провести AI-анализ договора", type="primary", width="stretch",
            disabled=not st.session_state.contract_text,
        ):
            with st.spinner("Анализируем договор, сверяем с базой прецедентов и эталонами..."):
                try:
                    report = ContractAnalyzer().analyze(st.session_state.contract_text, role)
                except (ValueError, RuntimeError, OSError, ChromaError) as exc:
                    st.error(f"Не удалось выполнить анализ. {exc}")
                else:
                    st.session_state.report = report
                    st.session_state.docx_bytes = None

        st.divider()
        st.subheader("База знаний")
        if st.button("🔄 Переиндексировать базу знаний", width="stretch"):
            with st.spinner("Обновляем индекс ChromaDB..."):
                try:
                    counts = LegalVectorStore().index_knowledge_base(force_reindex=True)
                except (ValueError, RuntimeError, OSError, ChromaError) as exc:
                    st.error(f"Не удалось обновить базу знаний. {exc}")
                else:
                    st.success(f"Записано фрагментов: {sum(counts.values())}.")
        try:
            counts = LegalVectorStore.knowledge_base_counts()
        except (ValueError, RuntimeError, OSError, ChromaError):
            st.warning("Не удалось прочитать статус индекса ChromaDB.")
        else:
            if sum(counts.values()):
                st.caption(
                    f"Эталоны: {counts['contract_templates']} · "
                    f"Обзоры рисков: {counts['legal_precedents']}"
                )
            else:
                st.caption("Индекс пуст. База заполняется при первом анализе.")
        st.warning(
            "⚠️ Учебный/демонстрационный проект. Результаты анализа носят справочный "
            "характер и не заменяют консультацию квалифицированного юриста."
        )


def _render_summary(report: ContractAnalysisReport) -> None:
    """Показать резюме, ключевые условия и нестандартные пункты."""
    st.text(report.executive_summary)
    st.subheader("Ключевые условия")
    if report.key_conditions:
        st.dataframe(
            [
                {"Условие": item.title, "Содержание": item.value, "Пункт": item.clause_ref}
                for item in report.key_conditions
            ],
            hide_index=True,
        )
    else:
        st.caption("Ключевые условия не выделены.")
    st.subheader("Нестандартные пункты")
    if report.non_standard_clauses:
        for clause in report.non_standard_clauses:
            st.text(f"• {clause}")
    else:
        st.caption("Нестандартные пункты не выявлены.")


def _render_risks(report: ContractAnalysisReport) -> None:
    """Показать риски выбранного уровня и формулировки для копирования."""
    level = st.selectbox("Уровень риска", ["Все", *_LEVELS.values()], key="risk_filter")
    risks = [risk for risk in report.risks if level == "Все" or _LEVELS[risk.level] == level]
    if not risks:
        st.info("Для выбранного уровня рисков нет.")
    for risk in risks:
        with st.container(border=True):
            st.html(
                f'{_badge(risk.level, risk.title)}<div class="risk-quote">'
                f'<mark class="highlight-{risk.level}">{escape(risk.exact_quote)}</mark></div>'
            )
            st.text(risk.explanation)
            if risk.precedent_ref:
                st.caption(f"Источник из базы знаний: {risk.precedent_ref}")
                for fragment in report.retrieved_precedents:
                    if fragment["source"] == risk.precedent_ref:
                        link = _SOURCE_URL.search(fragment["text"])
                        if link:
                            st.link_button("Открыть источник", link.group().rstrip(".,;)"))
                            break
            st.caption("Редакция для протокола разногласий")
            st.code(risk.recommendation, language=None)


def _render_sources(title: str, fragments: Sequence[RetrievedFragment]) -> None:
    """Показать происхождение, релевантность и текст каждого фрагмента."""
    st.subheader(title)
    if not fragments:
        st.caption("Релевантных фрагментов не найдено.")
    for index, fragment in enumerate(fragments, start=1):
        score = fragment.get("score")
        label = f"score {score:.3f}" if score is not None else "score недоступен"
        with st.expander(f"{index}. {fragment['source']} · {label}"):
            st.text(fragment["text"])


def _render_report(report: ContractAnalysisReport, file_name: str) -> None:
    """Показать метрики, экспорт и вкладки готового отчёта."""
    st.caption("Тип договора")
    st.text(report.contract_type)
    st.caption("Общий уровень риска")
    st.html(_badge(report.overall_risk_score))
    counts = Counter(risk.level for risk in report.risks)
    for column, (level, label) in zip(st.columns(3), _LEVELS.items(), strict=True):
        column.metric(f"{label} риск", counts[level])

    if st.session_state.docx_bytes is None:
        try:
            st.session_state.docx_bytes = generate_docx_report(report, file_name)
        except (ValueError, OSError):
            st.error("Не удалось подготовить файл заключения DOCX.")
    if st.session_state.docx_bytes is not None:
        clean_name = re.sub(r"[^\w.-]+", "_", Path(file_name).stem).strip("._") or "договор"
        st.download_button(
            "📥 Скачать юридическое заключение (.docx)",
            data=st.session_state.docx_bytes,
            file_name=f"AI_Анализ_{clean_name}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            on_click="ignore",
        )

    summary, risks, comparison, sources = st.tabs(
        ["📋 Резюме и условия", "⚠️ Риски и рекомендации", "⚖️ Сравнение с эталоном", "📚 База прецедентов (RAG)"]
    )
    with summary:
        _render_summary(report)
    with risks:
        _render_risks(report)
    with comparison:
        if report.template_comparison:
            for item in report.template_comparison:
                with st.container(border=True):
                    st.text(item.aspect)
                    st.caption(_STATUSES[item.status])
                    st.text(f"Договор: {item.contract_value}")
                    st.text(f"Эталон: {item.template_value}")
                    st.text(item.comment)
        else:
            st.info("Нет данных для сравнения с эталоном.")
    with sources:
        _render_sources("Обзоры юридических рисков", report.retrieved_precedents)
        _render_sources("Эталонные шаблоны", report.retrieved_templates)


def main() -> None:
    """Показать двухпанельный интерфейс и обработать действия пользователя."""
    st.set_page_config(page_title="AI-ассистент юридического анализа", page_icon="⚖️", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    _sidebar()
    st.title("AI-ассистент юридического анализа")
    text = st.session_state.contract_text
    file_name = st.session_state.file_name
    report = st.session_state.report
    contract_panel, analysis_panel = st.columns([1.05, 1.15], gap="large")
    with contract_panel:
        st.subheader("📄 Текст договора")
        with st.container(height=680, border=True):
            if st.session_state.document_error:
                st.error(st.session_state.document_error)
            elif text:
                st.caption(file_name)
                st.html(render_contract_html(text, report.risks if report else ()))
            else:
                st.info("Загрузите договор или выберите тестовый пример в боковой панели.")
    with analysis_panel:
        st.subheader("📊 Результаты AI-анализа")
        if report:
            _render_report(report, file_name)
        else:
            st.info("Выберите сторону и нажмите «Провести AI-анализ договора» в боковой панели.")
            if text:
                paragraphs = sum(bool(part.strip()) for part in _PARAGRAPH_BREAK.split(text))
                symbols, blocks = st.columns(2)
                symbols.metric("Символы", len(text))
                blocks.metric("Абзацы", paragraphs)


if __name__ == "__main__":
    main()
