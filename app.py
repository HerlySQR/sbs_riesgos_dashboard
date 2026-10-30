from __future__ import annotations

from datetime import datetime, date
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
import hashlib
import re
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader

SBS_URL = "https://www.sbs.gob.pe/app/iece/paginas/MostrarResumenClasificaciones.aspx"
BASE_URL = "https://www.sbs.gob.pe"
DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
CURRENT_CSV = DATA_DIR / "captura_actual.csv"
HISTORY_CSV = DATA_DIR / "historial_clasificaciones.csv"
REPORTS_CSV = DATA_DIR / "informes_descubiertos.csv"
DOCS_CSV = DATA_DIR / "clasificaciones_desde_informes.csv"
USER_AGENT = "Mozilla/5.0 (compatible; SBS-Risk-History-Dashboard/2.0)"

st.set_page_config(page_title="Histórico de riesgos SBS", page_icon="📈", layout="wide")
st.title("Clasificaciones de riesgo SBS — histórico y dashboard")
st.caption("Fuente oficial SBS. Las calificaciones son opiniones de las empresas clasificadoras y no constituyen una calificación otorgada por la SBS.")

DATA_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def http_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s

def clean_text(x):
    return re.sub(r"\s+", " ", str(x or "")).replace("\xa0", " ").strip()

def read_html_tables(html):
    try:
        return pd.read_html(StringIO(html))
    except ValueError:
        return []

def fetch_current(session):
    response = session.get(SBS_URL, timeout=45)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    soup = BeautifulSoup(response.text, "lxml")
    tables = read_html_tables(response.text)
    if not tables:
        raise RuntimeError("No se detectaron tablas HTML en la página de clasificaciones.")
    # Find the table with entity type and classifier names.
    chosen = None
    for table in tables:
        flat_cols = " ".join(map(str, table.columns)).lower()
        if "entidad" in flat_cols and len(table.columns) >= 3:
            chosen = table.copy()
            break
    if chosen is None:
        # Fallback: inspect first rows in case headers are multi-row.
        for table in tables:
            text = " ".join(map(str, table.head(4).values.flatten())).lower()
            if "entidad" in text and len(table.columns) >= 3:
                chosen = table.copy()
                break
    if chosen is None:
        raise RuntimeError("La estructura HTML de la SBS cambió; no se identificó la tabla principal.")

    if isinstance(chosen.columns, pd.MultiIndex):
        chosen.columns = [
            " - ".join(clean_text(p) for p in tup if clean_text(p) and not clean_text(p).lower().startswith("unnamed"))
            for tup in chosen.columns
        ]
    else:
        chosen.columns = [clean_text(c) for c in chosen.columns]

    type_col = next((c for c in chosen.columns if "tipo" in c.lower()), None)
    entity_col = next((c for c in chosen.columns if "entidad" in c.lower()), None)
    if entity_col is None:
        # Sometimes first columns are unnamed after a dynamic page render.
        entity_col = chosen.columns[1] if len(chosen.columns) > 1 else chosen.columns[0]
    if type_col is None:
        type_col = chosen.columns[0] if chosen.columns[0] != entity_col else None

    captured_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    for _, row in chosen.iterrows():
        entity = clean_text(row.get(entity_col, ""))
        if not entity or entity.lower() in {"nan", "entidad", "none"}:
            continue
        entity_type = clean_text(row.get(type_col, "No especificado")) if type_col else "No especificado"
        for col in chosen.columns:
            if col in {entity_col, type_col}:
                continue
            rating = clean_text(row.get(col, ""))
            if not rating or rating.lower() in {"nan", "none", "-"}:
                continue
            # Avoid treating arbitrary table notes as ratings.
            if len(rating) > 30:
                continue
            rows.append({
                "fecha_observacion": captured_at[:10],
                "fecha_informe": None,
                "fecha_descarga": captured_at,
                "tipo_entidad": entity_type,
                "entidad": entity,
                "clasificadora": clean_text(col),
                "clasificacion": rating,
                "fuente_tipo": "captura_actual_sbs",
                "fuente_url": SBS_URL,
                "archivo_local": ""
            })
    current = pd.DataFrame(rows)
    if current.empty:
        raise RuntimeError("La tabla se descargó, pero no se pudieron extraer filas de clasificación.")
    return current, soup

def discover_report_links(soup):
    """Collect linked SBS/extranet report documents from the current results page."""
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(SBS_URL, a["href"])
        label = clean_text(a.get_text(" ", strip=True))
        parsed = urlparse(href)
        if not parsed.scheme.startswith("http"):
            continue
        # SBS classification links usually lead to the I E C E document endpoint.
        relevant = (
            "extranet.sbs.gob.pe/iece/" in href.lower()
            or "codclasificadora=" in href.lower()
            or "mostrarresumenclasificaciones" in href.lower()
        )
        if relevant:
            q = parse_qs(parsed.query)
            links.append({
                "etiqueta": label,
                "url": href,
                "cod_clasificadora": (q.get("codClasificadora") or [""])[0],
                "cod_periodo": (q.get("codPeriodo") or [""])[0],
                "num_archivo": (q.get("numArchivo") or [""])[0],
                "num_version": (q.get("numVersion") or [""])[0],
            })
    return pd.DataFrame(links).drop_duplicates(subset=["url"]) if links else pd.DataFrame(
        columns=["etiqueta", "url", "cod_clasificadora", "cod_periodo", "num_archivo", "num_version"]
    )

def extract_period_from_text(text, url=""):
    # Prefer an explicit report period/month in extracted PDF text.
    patterns = [
        r"(?:INFORME\s+DE\s+CLASIFICACI[ÓO]N|CLASIFICACI[ÓO]N)\s*[-–—:]?\s*(?:al\s+)?(Diciembre|Junio|Marzo|Setiembre|Septiembre)\s+(20\d{2})",
        r"\b(Junio|Diciembre|Marzo|Setiembre|Septiembre)\s+(20\d{2})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text[:25000], flags=re.I)
        if m:
            month = m.group(1).lower()
            year = int(m.group(2))
            month_num = {"marzo": 3, "junio": 6, "setiembre": 9, "septiembre": 9, "diciembre": 12}.get(month)
            if month_num:
                return date(year, month_num, 1).isoformat()
    # Endpoint period often has YYYYMM, e.g. codPeriodo=202601.
    m = re.search(r"codPeriodo=(20\d{2})(0[1-9]|1[0-2])", url, re.I)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1).isoformat()
    return ""

def parse_rating_candidates(text):
    """Extract likely rating tokens from PDF text; intentionally conservative."""
    pattern = r"(?<![A-Z0-9])(?:AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|CCC[+-]?|CC|C|D|RET|EQL|PE\d)(?![A-Z0-9])"
    found = [m.group(0).upper() for m in re.finditer(pattern, text.upper())]
    return found

def extract_pdf_report(session, report, max_bytes=20_000_000):
    url = report["url"]
    response = session.get(url, timeout=60, stream=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    data = response.content
    if len(data) > max_bytes:
        raise RuntimeError("Documento supera el límite de tamaño configurado.")
    if not (data.startswith(b"%PDF") or "pdf" in content_type):
        raise RuntimeError("El enlace no devolvió un PDF.")
    digest = hashlib.sha256(data).hexdigest()[:16]
    name = f"{digest}.pdf"
    path = REPORT_DIR / name
    path.write_bytes(data)
    reader = PdfReader(BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)
    period = extract_period_from_text(text, url)
    return {
        "url": url,
        "archivo_local": str(path),
        "sha256_corto": digest,
        "fecha_informe": period,
        "texto": text,
        "paginas": len(reader.pages),
        "texto_extraido": bool(text.strip()),
        "error": ""
    }

def load_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path, dtype=str).fillna("")
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_frame(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def append_unique(old, new, keys):
    if old is None or old.empty:
        out = new.copy()
    elif new is None or new.empty:
        out = old.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
    if not out.empty:
        for key in keys:
            if key not in out.columns:
                out[key] = ""
        out = out.drop_duplicates(subset=keys, keep="last")
    return out

def build_report_records(session, links_df, max_reports):
    old_reports = load_csv(REPORTS_CSV)
    old_urls = set(old_reports.get("url", pd.Series(dtype=str)).astype(str))
    candidates = links_df[~links_df["url"].isin(old_urls)].head(max_reports)
    report_rows, extracted_rows, errors = [], [], []
    current = load_csv(CURRENT_CSV)
    for _, report in candidates.iterrows():
        try:
            parsed = extract_pdf_report(session, report)
            report_rows.append({
                **report.to_dict(),
                "fecha_informe": parsed["fecha_informe"],
                "archivo_local": parsed["archivo_local"],
                "sha256_corto": parsed["sha256_corto"],
                "paginas": parsed["paginas"],
                "texto_extraido": parsed["texto_extraido"],
                "estado": "descargado"
            })
            # PDFs are source evidence. Do not pretend a bare token found in narrative text
            # is a verified entity rating. Produce review candidates for human confirmation.
            txt = parsed["texto"]
            ratings = parse_rating_candidates(txt)
            if txt.strip():
                entity = ""
                # Best-effort entity matching against current entity list.
                entities = sorted(current.get("entidad", pd.Series(dtype=str)).dropna().unique(), key=len, reverse=True)
                for candidate_entity in entities:
                    if len(candidate_entity) >= 5 and candidate_entity.lower() in txt.lower():
                        entity = candidate_entity
                        break
                unique_ratings = list(dict.fromkeys(ratings))
                extracted_rows.append({
                    "fecha_informe": parsed["fecha_informe"],
                    "entidad_candidata": entity,
                    "clasificadora_candidata": clean_text(report.get("etiqueta", "")),
                    "clasificaciones_detectadas_texto": " | ".join(unique_ratings[:30]),
                    "url": report["url"],
                    "archivo_local": parsed["archivo_local"],
                    "requiere_revision": "Sí",
                    "nota": "Candidatos detectados en el texto completo; no son registros de clasificación verificados automáticamente."
                })
        except Exception as exc:
            errors.append(f"{report['url']}: {exc}")
            report_rows.append({
                **report.to_dict(),
                "fecha_informe": "",
                "archivo_local": "",
                "sha256_corto": "",
                "paginas": "",
                "texto_extraido": False,
                "estado": f"error: {exc}"
            })
        time.sleep(0.25)
    if report_rows:
        old_reports = append_unique(old_reports, pd.DataFrame(report_rows), ["url"])
        save_frame(old_reports, REPORTS_CSV)
    if extracted_rows:
        old_extracted = load_csv(DOCS_CSV)
        old_extracted = append_unique(old_extracted, pd.DataFrame(extracted_rows), ["url"])
        save_frame(old_extracted, DOCS_CSV)
    return len(report_rows), errors

def load_history():
    df = load_csv(HISTORY_CSV)
    if df.empty:
        return df
    for c in ["fecha_observacion", "fecha_informe", "fecha_descarga", "tipo_entidad", "entidad",
              "clasificadora", "clasificacion", "fuente_tipo", "fuente_url", "archivo_local"]:
        if c not in df:
            df[c] = ""
    return df

with st.sidebar:
    st.header("Actualizar información")
    st.write("La captura actual incluye todas las categorías visibles en la tabla SBS, entre ellas bancos, financieras, cajas, empresas de créditos y seguros.")
    if st.button("1. Descargar clasificación actual", type="primary", use_container_width=True):
        try:
            session = http_session()
            with st.spinner("Descargando la tabla actual de la SBS..."):
                current, soup = fetch_current(session)
                save_frame(current, CURRENT_CSV)
                old = load_history()
                history = append_unique(old, current, [
                    "fecha_observacion", "tipo_entidad", "entidad", "clasificadora", "clasificacion", "fuente_tipo"
                ])
                save_frame(history, HISTORY_CSV)
                links = discover_report_links(soup)
                old_links = load_csv(REPORTS_CSV)
                if not links.empty and not old_links.empty:
                    links = links[~links["url"].isin(set(old_links["url"]))]
                if not links.empty:
                    st.session_state["pending_links"] = links.to_dict("records")
            st.success(f"Captura actual guardada: {len(current)} registros.")
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo actualizar la tabla: {e}")

    st.divider()
    st.write("**Informes enlazados en la página actual**")
    max_reports = st.number_input("Máximo de PDFs por ejecución", min_value=1, max_value=100, value=20, step=5)
    if st.button("2. Descargar y analizar PDFs detectados", use_container_width=True):
        try:
            session = http_session()
            # Refresh page so links are current.
            response = session.get(SBS_URL, timeout=45)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            links = discover_report_links(soup)
            old_reports = load_csv(REPORTS_CSV)
            if not old_reports.empty:
                links = links[~links["url"].isin(set(old_reports["url"]))]
            if links.empty:
                st.warning("No se encontraron PDFs nuevos en los enlaces de la página actual. La página podría requerir elegir otro período en sus filtros.")
            else:
                with st.spinner(f"Descargando hasta {int(max_reports)} documentos..."):
                    count, errors = build_report_records(session, links, int(max_reports))
                st.success(f"Procesados {count} enlaces nuevos.")
                if errors:
                    st.warning(f"{len(errors)} documentos no se pudieron procesar. Revisa el CSV de informes.")
            st.rerun()
        except Exception as e:
            st.error(f"Error al procesar documentos: {e}")

    st.caption("Los enlaces encontrados corresponden a los documentos accesibles desde el resultado actual. Para periodos antiguos puede ser necesario cambiar el filtro de periodo en la página SBS y volver a ejecutar la descarga.")

history = load_history()
current = load_csv(CURRENT_CSV)
reports = load_csv(REPORTS_CSV)
pdf_candidates = load_csv(DOCS_CSV)

if history.empty:
    st.info("Aún no hay datos locales. Pulsa **Descargar clasificación actual** en el panel izquierdo para iniciar la base histórica.")
    st.markdown(f"Fuente oficial: [{SBS_URL}]({SBS_URL})")
    st.stop()

for c in ["fecha_observacion", "fecha_informe", "fecha_descarga"]:
    if c in history:
        history[c] = pd.to_datetime(history[c], errors="coerce")

st.subheader("Resumen")
latest_capture = history["fecha_observacion"].max()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Entidades", history["entidad"].nunique())
m2.metric("Clasificadoras", history["clasificadora"].nunique())
m3.metric("Registros guardados", len(history))
m4.metric("Documentos descargados", len(reports))

st.subheader("Explorar las clasificaciones")
col1, col2, col3 = st.columns(3)
types = ["Todos"] + sorted(history["tipo_entidad"].replace("", "No especificado").unique().tolist())
type_filter = col1.selectbox("Tipo de entidad", types)
entities_all = sorted(history["entidad"].unique())
if type_filter != "Todos":
    entities_all = sorted(history.loc[history["tipo_entidad"].replace("", "No especificado") == type_filter, "entidad"].unique())
entity_filter = col2.selectbox("Entidad", ["Todas"] + entities_all)
classifier_options = sorted(history["clasificadora"].unique())
classifier_filter = col3.multiselect("Clasificadora", classifier_options, default=classifier_options)

filtered = history.copy()
if type_filter != "Todos":
    filtered = filtered[filtered["tipo_entidad"].replace("", "No especificado") == type_filter]
if entity_filter != "Todas":
    filtered = filtered[filtered["entidad"] == entity_filter]
if classifier_filter:
    filtered = filtered[filtered["clasificadora"].isin(classifier_filter)]

st.dataframe(
    filtered.sort_values(["entidad", "clasificadora", "fecha_observacion"], na_position="last"),
    use_container_width=True, hide_index=True
)

st.subheader("Serie temporal")
st.write("La serie distingue la fecha de observación de la fecha oficial del informe. Los puntos de captura actual reflejan cuándo el programa vio la calificación, no necesariamente cuándo fue emitida.")
series = filtered.copy()
series["fecha_grafico"] = series["fecha_informe"].fillna(series["fecha_observacion"])
series["fecha_grafico"] = pd.to_datetime(series["fecha_grafico"], errors="coerce")
series = series.dropna(subset=["fecha_grafico"])
# Only plot when the same entity/classifier has at least two observations.
if series.empty:
    st.info("No hay fechas válidas para graficar.")
else:
    target_entity = entity_filter if entity_filter != "Todas" else st.selectbox("Entidad para la serie", sorted(series["entidad"].unique()))
    graph = series[series["entidad"] == target_entity].copy()
    if graph.empty:
        st.info("No hay observaciones para esa entidad.")
    else:
        # Do not force a numeric order across different classifier methodologies.
        graph = graph.sort_values("fecha_grafico")
        fig = px.scatter(
            graph, x="fecha_grafico", y="clasificacion", color="clasificadora",
            symbol="fuente_tipo", hover_data=["tipo_entidad", "fecha_informe", "fecha_observacion", "fuente_url"],
            title=f"Clasificaciones observadas — {target_entity}",
            labels={"fecha_grafico": "Fecha (informe cuando se conoce; si no, captura)",
                    "clasificacion": "Calificación publicada", "clasificadora": "Clasificadora",
                    "fuente_tipo": "Origen"}
        )
        fig.update_traces(marker={"size": 11})
        fig.update_layout(legend_title_text="Clasificadora / origen")
        st.plotly_chart(fig, use_container_width=True)
        if graph["fecha_grafico"].nunique() < 2:
            st.caption("Solo hay una fecha para esta entidad. La serie se ampliará cuando se recuperen informes históricos o se guarden nuevas capturas.")

st.subheader("Informes históricos: cobertura y revisión")
if reports.empty:
    st.info("Todavía no hay informes descargados. Usa el botón de análisis de PDFs en el panel izquierdo.")
else:
    st.dataframe(reports, use_container_width=True, hide_index=True)
if not pdf_candidates.empty:
    st.warning("Las detecciones textuales de los PDF son candidatos de extracción, no datos históricos verificados. Revisa el documento original antes de usarlos como observaciones.")
    st.dataframe(pdf_candidates, use_container_width=True, hide_index=True)

st.subheader("Exportar")
for path, label in [
    (HISTORY_CSV, "Descargar histórico CSV"),
    (REPORTS_CSV, "Descargar inventario de informes CSV"),
    (DOCS_CSV, "Descargar candidatos extraídos de PDF CSV"),
]:
    if path.exists():
        st.download_button(label, data=path.read_bytes(), file_name=path.name, mime="text/csv", key=f"dl_{path.name}")

st.markdown(f"**Fuente:** [Clasificaciones e informes semestrales de la SBS]({SBS_URL})")
st.caption("Importante: esta versión recupera PDFs enlazados en la página consultada y extrae texto para revisión. No afirma haber reconstruido automáticamente todo el archivo histórico de la SBS. Los PDFs escaneados pueden requerir OCR y las fechas/calificaciones deben validarse.")
