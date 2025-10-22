import os
import json
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
import pg8000
from dotenv import load_dotenv, find_dotenv
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from urllib.parse import urlparse
import pandas as pd
from html import unescape

# Cargar variables de entorno desde .env
load_dotenv(find_dotenv(), override=True)

def extract_domain(url: str) -> str:
    try:
        if not url:
            return ""
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

# items: list[dict] where each dict has at least keys: 'url', 'relevant', 'created_at'
def aggregate_by_journal(items: list[dict]) -> list[dict]:
    stats: dict[str, dict] = {}
    for it in items:
        domain = extract_domain(it.get("url", "")) or "(sin dominio)"
        s = stats.setdefault(domain, {"count": 0, "relevant_count": 0, "min_year": None, "max_year": None})
        s["count"] += 1
        if it.get("relevant"):
            s["relevant_count"] += 1
        created_at = it.get("created_at")
        try:
            if created_at:
                year = created_at.year if hasattr(created_at, "year") else int(str(created_at)[:4])
                s["min_year"] = year if s["min_year"] is None else min(s["min_year"], year)
                s["max_year"] = year if s["max_year"] is None else max(s["max_year"], year)
        except Exception:
            pass
    rows = [
        {
            "domain": d,
            "count": v["count"],
            "relevant_count": v["relevant_count"],
            "min_year": v["min_year"],
            "max_year": v["max_year"],
        }
        for d, v in stats.items()
    ]
    rows.sort(key=lambda r: (r["relevant_count"], r["count"]), reverse=True)
    return rows

st.set_page_config(page_title="Asistente de Investigación", page_icon="📚", layout="wide")

# Conexión a Postgres

def build_db_url():
    host = os.getenv("DATABASE_HOST")
    port = os.getenv("DATABASE_PORT")
    dbname = os.getenv("DATABASE_NAME")
    user = os.getenv("DATABASE_USERNAME")
    password = os.getenv("DATABASE_PASSWORD")
    ssl = os.getenv("DATABASE_SSL", "false").lower()
    if not all([host, port, dbname, user, password]):
        return None
    base = f"postgresql+pg8000://{user}:{password}@{host}:{port}/{dbname}"
    if ssl in ("false", "0", "no"):
        return base + "?sslmode=disable"
    return base

@st.cache_resource
def get_engine():
    # Conservado por compatibilidad, ya no usado con pg8000
    return None

# Consultar artículos como lista de dicts

def fetch_articles(limit=500):
    host = os.getenv("DATABASE_HOST")
    port = os.getenv("DATABASE_PORT")
    dbname = os.getenv("DATABASE_NAME")
    user = os.getenv("DATABASE_USERNAME")
    password = os.getenv("DATABASE_PASSWORD")
    if not all([host, port, dbname, user, password]):
        return []
    try:
        with pg8000.connect(user=user, password=password, host=host, port=int(port), database=dbname) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, title, authors, year, url, doi, abstract, source, relevant, summary, findings, created_at FROM articles ORDER BY created_at DESC LIMIT %s", (limit,))
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []

# Disparo del flujo n8n

def trigger_n8n(topic: str, date_from: str, date_to: str):
    webhook_url = os.getenv("N8N_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return {"error": "No está configurada N8N_WEBHOOK_URL en .env"}
    try:
        resp = requests.post(webhook_url, json={"topic": topic, "from": date_from, "to": date_to}, timeout=60)
        if resp.status_code >= 400:
            return {"error": f"n8n respondió {resp.status_code}", "body": resp.text}
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# Generar PDF

def generate_pdf(items: list) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Título
    title = Paragraph("Reporte de artículos", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Estadísticas
    total = len(items)
    relevantes = sum(1 for i in items if i.get("relevant"))
    por_fuente = {}
    for i in items:
        src = i.get("source") or ""
        por_fuente[src] = por_fuente.get(src, 0) + 1
    
    stats_text = f"Total artículos: {total}<br/>Relevantes: {relevantes}<br/>Por fuente: {json.dumps(por_fuente, ensure_ascii=False)}"
    stats = Paragraph(stats_text, styles['Normal'])
    story.append(stats)
    story.append(Spacer(1, 12))
    
    # Listado
    listado_title = Paragraph("Listado", styles['Heading2'])
    story.append(listado_title)
    story.append(Spacer(1, 12))
    
    for row in items:
        titulo = (row.get("title") or "")[:160]
        resumen = (row.get("summary") or "")[:300]
        url = row.get("url") or row.get("doi") or ""
        rel = "Sí" if row.get("relevant") else "No"
        year = row.get("year") or ""
        
        article_text = f"<b>[{year}] {titulo}</b> (Relevante: {rel})<br/>{resumen}<br/><i>{url}</i>"
        article = Paragraph(article_text, styles['Normal'])
        story.append(article)
        story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# UI
st.title("📚 Asistente de Investigación (Streamlit + n8n + Postgres)")

def ensure_articles_table_exists() -> bool:
    host = os.getenv("DATABASE_HOST")
    port = os.getenv("DATABASE_PORT")
    dbname = os.getenv("DATABASE_NAME")
    user = os.getenv("DATABASE_USERNAME")
    password = os.getenv("DATABASE_PASSWORD")
    if not all([host, port, dbname, user, password]):
        return False
    try:
        with pg8000.connect(user=user, password=password, host=host, port=int(port), database=dbname) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.articles')")
                exists = cur.fetchone()[0] is not None
                return exists
    except Exception:
        return False


def initialize_articles_table() -> tuple[bool, str]:
    host = os.getenv("DATABASE_HOST")
    port = os.getenv("DATABASE_PORT")
    dbname = os.getenv("DATABASE_NAME")
    user = os.getenv("DATABASE_USERNAME")
    password = os.getenv("DATABASE_PASSWORD")
    if not all([host, port, dbname, user, password]):
        return False, "Faltan variables de conexión en .env"
    sql_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db.sql'))
    try:
        with open(sql_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        statements = [s.strip() for s in sql_content.split(';') if s.strip()]
        with pg8000.connect(user=user, password=password, host=host, port=int(port), database=dbname) as conn:
            with conn.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)
            conn.commit()
        return True, "Tabla creada/actualizada"
    except Exception as e:
        return False, str(e)

def list_tables() -> list[dict]:
    host = os.getenv("DATABASE_HOST")
    port = os.getenv("DATABASE_PORT")
    dbname = os.getenv("DATABASE_NAME")
    user = os.getenv("DATABASE_USERNAME")
    password = os.getenv("DATABASE_PASSWORD")
    if not all([host, port, dbname, user, password]):
        return []
    try:
        with pg8000.connect(user=user, password=password, host=host, port=int(port), database=dbname) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT schemaname, tablename
                    FROM pg_catalog.pg_tables
                    WHERE schemaname NOT IN ('pg_catalog','information_schema')
                    ORDER BY schemaname, tablename
                    """
                )
                rows = cur.fetchall()
                return [{"schema": r[0], "table": r[1]} for r in rows]
    except Exception:
        return []


with st.sidebar:
    st.header("Conexión a BD")
    db_url = build_db_url()
    if db_url:
        st.success("Postgres configurado")
    else:
        st.error("Faltan variables de BD en .env")

    # Estado de la tabla articles y acción de inicialización
    exists = ensure_articles_table_exists()
    st.caption("Tabla articles")
    if exists:
        st.success("Existe")
    else:
        st.warning("No existe")
    cta1, cta2 = st.columns(2)
    with cta1:
        if st.button("Crear tabla articles"):
            ok, msg = initialize_articles_table()
            if ok:
                st.success(msg)
            else:
                st.error(f"Error: {msg}")
    with cta2:
        if st.button("Aplicar db.sql (migración)"):
            ok, msg = initialize_articles_table()
            if ok:
                st.success(msg)
            else:
                st.error(f"Error: {msg}")

    st.caption("Tablas en BD")
    tbls = list_tables()
    if tbls:
        st.dataframe(tbls, use_container_width=True)
    else:
        st.info("No se encontraron tablas visibles en este esquema.")

    st.header("Flujo n8n")
    webhook_url = os.getenv("N8N_WEBHOOK_URL")
    if webhook_url:
        st.success("Webhook configurado")
        st.caption("Destino del webhook")
        st.code(webhook_url)
    else:
        st.warning("Configura N8N_WEBHOOK_URL para ejecutar el flujo desde la app")

tab1, tab2, tab3 = st.tabs(["Buscar/Ejecutar", "Analítica", "Reporte PDF"]) 

def prepare_articles_table(items: list[dict]):
    import pandas as _pd
    df = _pd.DataFrame(items)
    if df.empty:
        return df
    # Añadir dominio y seleccionar columnas amigables
    if 'url' in df.columns:
        df['domain'] = df['url'].apply(lambda u: extract_domain(u) if isinstance(u, str) else '')
    cols = [c for c in ['title','authors','year','domain','doi','url','relevant','created_at'] if c in df.columns]
    df = df[cols]
    # Truncar textos largos para vista compacta
    if 'title' in df.columns:
        df['title'] = df['title'].apply(lambda t: (t[:90] + '…') if isinstance(t, str) and len(t) > 90 else t)
    if 'authors' in df.columns:
        df['authors'] = df['authors'].apply(lambda a: (a[:80] + '…') if isinstance(a, str) and len(a) > 80 else a)
    return df
with tab1:
    st.subheader("Buscar artículos y ejecutar flujo n8n")
    col1, col2, col3 = st.columns(3)
    with col1:
        topic = st.text_input("Tema a buscar", value="biodiversity")
    with col2:
        default_from = (datetime.now(timezone.utc) - timedelta(days=int(os.getenv("FLOW_DEFAULT_FROM_DAYS", "30")))).date()
        date_from = st.date_input("Desde", value=default_from)
    with col3:
        date_to = st.date_input("Hasta", value=datetime.now(timezone.utc).date())

    # Acciones apiladas para ocupar todo el ancho
    if st.button("Ejecutar flujo n8n"):
        res = trigger_n8n(topic, date_from.strftime('%Y-%m-%d'), date_to.strftime('%Y-%m-%d'))
        if "error" in res:
            st.error(f"Error: {res['error']}")
            if res.get("body"):
                st.code(res["body"]) 
        else:
            st.success("Flujo ejecutado")
            st.json(res)

    if st.button("Actualizar artículos desde BD"):
        items = fetch_articles()
        if not items:
            st.warning("No hay artículos en BD aún")
        else:
            st.success(f"Cargados {len(items)} artículos")
            df = prepare_articles_table(items)
            # Filtro de texto
            q = st.text_input("Buscar por título/autor/dominio", key="tbl_search_tab1")
            if q:
                ql = q.lower()
                def _match(row):
                    return any(str(row.get(col, '')).lower().find(ql) >= 0 for col in ['title','authors','domain'])
                df = df[df.apply(_match, axis=1)]
            # Tabla a ancho completo bajo los botones
            st.dataframe(
                df,
                use_container_width=True,
                height=420,
                column_config={
                    "title": st.column_config.TextColumn("Título", width="medium"),
                    "authors": st.column_config.TextColumn("Autores", width="medium"),
                    "year": st.column_config.NumberColumn("Año"),
                    "domain": st.column_config.TextColumn("Revista/Dominio", width="medium"),
                    "doi": st.column_config.LinkColumn("DOI", display_text="Abrir"),
                    "url": st.column_config.LinkColumn("URL", display_text="Abrir"),
                    "relevant": st.column_config.CheckboxColumn("Relevante"),
                    "created_at": st.column_config.DatetimeColumn("Creado", format="YYYY-MM-DD"),
                }
            )
            st.caption(f"{len(df)} artículos mostrados")
            # Detalle opcional
            if not df.empty and 'title' in df.columns:
                sel = st.selectbox("Ver detalle", list(df['title']))
                # Buscar datos completos en items
                full = next((i for i in items if (i.get('title') or '').startswith(sel[:-1]) or i.get('title') == sel), None)
                if full:
                    with st.expander("Detalle del artículo"):
                        st.markdown(f"**{full.get('title','')}**")
                        st.write(f"Autores: {full.get('authors','')}")
                        st.write(f"Año: {full.get('year','')}")
                        st.write(f"Dominio: {extract_domain(full.get('url',''))}")
                        if full.get('doi'):
                            st.write(f"DOI: {full.get('doi')}")
                        if full.get('url'):
                            st.write(f"URL: {full.get('url')}")
                        if full.get('abstract'):
                            st.markdown(unescape(str(full.get('abstract'))), unsafe_allow_html=True)
                        if full.get('summary'):
                            st.markdown("Resumen:")
                            st.markdown(unescape(str(full.get('summary'))), unsafe_allow_html=True)

with tab2:
    st.subheader("Estadísticas y gráficos")
    items = fetch_articles()
    if not items:
        st.info("No hay datos. Ejecuta el flujo n8n o actualiza desde BD en la pestaña anterior.")
    else:
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Artículos", len(items))
        with colB:
            st.metric("Relevantes", sum(1 for i in items if i.get("relevant")))
        with colC:
            years = {i.get("year") for i in items if i.get("year") is not None}
            st.metric("Años distintos", len(years))

        st.write("Distribución por fuente")
        src_counts = {}
        for i in items:
            src = i.get("source") or ""
            src_counts[src] = src_counts.get(src, 0) + 1
        if src_counts:
            fig = px.bar(x=list(src_counts.keys()), y=list(src_counts.values()), 
                        labels={'x': 'Fuente', 'y': 'Cantidad'}, height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.write("Revistas (por dominio de URL)")
        journals = aggregate_by_journal(items)
        if journals:
            top = journals[:15]
            figj = px.bar(x=[j["domain"] for j in top], y=[j["count"] for j in top], labels={'x': 'Dominio', 'y': 'Artículos'}, height=300)
            st.plotly_chart(figj, use_container_width=True)
            st.dataframe(journals, use_container_width=True)

        st.write("Relevantes por año")
        rel_year_map = {}
        for i in items:
            y = i.get("year")
            if y is None:
                continue
            key = (y, bool(i.get("relevant")))
            rel_year_map[key] = rel_year_map.get(key, 0) + 1
        
        if rel_year_map:
            years = []
            relevant_counts = []
            not_relevant_counts = []
            
            all_years = sorted(set(k[0] for k in rel_year_map.keys()))
            for year in all_years:
                years.append(str(year))
                relevant_counts.append(rel_year_map.get((year, True), 0))
                not_relevant_counts.append(rel_year_map.get((year, False), 0))
            
            fig = go.Figure(data=[
                go.Bar(name='Relevante', x=years, y=relevant_counts),
                go.Bar(name='No Relevante', x=years, y=not_relevant_counts)
            ])
            fig.update_layout(barmode='stack', height=300, xaxis_title='Año', yaxis_title='Cantidad')
            st.plotly_chart(fig, use_container_width=True)

        st.write("Tabla de artículos")
        st.dataframe(items, use_container_width=True)

with tab3:
    st.subheader("Generar reporte PDF")
    items = fetch_articles()
    if not items:
        st.info("No hay datos para el reporte.")
    else:
        only_relevant = st.checkbox("Solo relevantes", value=True)
        fallback_all = st.checkbox("Si no hay relevantes, incluir todos", value=True)
        filtered = [i for i in items if i.get("relevant")] if only_relevant else items
        if not filtered and fallback_all:
            filtered = items
        if st.button("Generar PDF"):
            if not filtered:
                st.warning("No hay artículos para el reporte (verifica filtros)")
            else:
                pdf_bytes = generate_pdf(filtered)
                st.success("Reporte generado")
                st.download_button(
                    label="Descargar PDF",
                    data=pdf_bytes,
                    file_name=f"reporte_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )


def extract_domain(url: str) -> str:
    try:
        if not url:
            return ""
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path  # handle plain domains
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

# items: list[dict] where each dict has at least keys: 'url', 'is_relevant', 'created_at'
def aggregate_by_journal(items: list[dict]) -> list[dict]:
    stats: dict[str, dict] = {}
    for it in items:
        domain = extract_domain(it.get("url", "")) or "(sin dominio)"
        s = stats.setdefault(domain, {"count": 0, "relevant_count": 0, "min_year": None, "max_year": None})
        s["count"] += 1
        if it.get("is_relevant"):
            s["relevant_count"] += 1
        # track year range if created_at present
        created_at = it.get("created_at")
        try:
            if created_at:
                year = created_at.year if hasattr(created_at, "year") else int(str(created_at)[:4])
                s["min_year"] = year if s["min_year"] is None else min(s["min_year"], year)
                s["max_year"] = year if s["max_year"] is None else max(s["max_year"], year)
        except Exception:
            pass
    # convert to list sorted by relevant_count desc, then count desc
    rows = [
        {
            "domain": d,
            "count": v["count"],
            "relevant_count": v["relevant_count"],
            "min_year": v["min_year"],
            "max_year": v["max_year"],
        }
        for d, v in stats.items()
    ]
    rows.sort(key=lambda r: (r["relevant_count"], r["count"]), reverse=True)
    return rows