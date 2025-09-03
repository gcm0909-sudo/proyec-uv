# -*- coding: utf-8 -*-
# Proyecto: Radiación UV Norte Grande de Chile (Open-Meteo, con fallback past_days)
# + Indicadores de Cobre desde mindicador.cl (USD/libra) y conversión a CLP/libra

import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
from datetime import date, timedelta, datetime, timezone

st.set_page_config(page_title="Radiación UV – Norte Grande", page_icon="☀️", layout="wide")

# -----------------------------
# Ciudades del Norte Grande
# -----------------------------
NORTE_GRANDE_CITIES = {
    "Arica": (-18.4783, -70.3126),
    "Iquique": (-20.2133, -70.1503),
    "Antofagasta": (-23.6500, -70.4000),
    "Calama": (-22.4550, -68.9290),
    "San Pedro de Atacama": (-22.9110, -68.2030),
    "Tocopilla": (-22.0887, -70.1936),
}

AYER = date.today() - timedelta(days=1)
DEFAULT_END = AYER
DEFAULT_START = DEFAULT_END - timedelta(days=365)

# =============================
# UTILIDADES GENERALES
# =============================
def _safe_json_get(url: str, params: dict | None = None, timeout: int = 60):
    """Wrapper simple para requests.get -> .json(), con manejo de errores."""
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"

# -----------------------------
# HISTÓRICO DIARIO UVI (archive -> fallback forecast+past_days)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_uv_daily_smart(lat: float, lon: float, start: date, end: date):
    """Primero intenta el endpoint 'archive'. Si viene vacío, usa 'forecast' con 'past_days'."""
    # --- 1) Intento: ARCHIVE ---
    url_arch = "https://archive-api.open-meteo.com/v1/archive"
    p_arch = {
        "latitude": lat, "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": "uv_index_max",
        "timezone": "auto",
    }
    data_arch, err_arch = _safe_json_get(url_arch, p_arch)

    def _to_df(d):
        if not d or "daily" not in d or "time" not in d["daily"]:
            return pd.DataFrame()
        daily = d["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "uv_index_max": pd.to_numeric(daily.get("uv_index_max", []), errors="coerce"),
        }).dropna(subset=["uv_index_max"])
        return df.sort_values("date").reset_index(drop=True)

    df_arch = _to_df(data_arch)
    if not df_arch.empty:
        return df_arch, None, {"source": "archive"}

    # --- 2) Fallback: FORECAST con past_days ---
    past_days = max((end - start).days + 1, 1)
    url_fc = "https://api.open-meteo.com/v1/forecast"
    p_fc = {
        "latitude": lat, "longitude": lon,
        "daily": "uv_index_max",
        "timezone": "auto",
        "past_days": past_days,
        "forecast_days": 1,
    }
    data_fc, err_fc = _safe_json_get(url_fc, p_fc)
    if err_fc and not err_arch:
        # Si falló forecast, al menos devuelve el error
        return pd.DataFrame(), f"Error consultando datos: {err_fc}", {"source": "none"}

    df_fc = _to_df(data_fc)
    if not df_fc.empty:
        mask = (df_fc["date"].dt.date >= start) & (df_fc["date"].dt.date <= end)
        df_fc = df_fc.loc[mask]
    if not df_fc.empty:
        return df_fc, None, {"source": "forecast(past_days)"}

    return pd.DataFrame(), "Sin datos de UVI para el rango/ubicación.", {"source": "empty"}

# -----------------------------
# PRONÓSTICO HORARIO UVI
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_uv_forecast_hourly(lat: float, lon: float, days: int = 5):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "uv_index",
        "timezone": "auto",
        "forecast_days": int(days),
    }
    data, err = _safe_json_get(url, params)
    if err:
        return pd.DataFrame(), f"Error pronóstico: {err}"

    if "hourly" not in data or "time" not in data["hourly"]:
        return pd.DataFrame(), "Sin datos horarios en la respuesta."

    h = data["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "uv_index": pd.to_numeric(h.get("uv_index", [np.nan] * len(h["time"])), errors="coerce")
    }).dropna(subset=["uv_index"])
    return df.sort_values("time").reset_index(drop=True), None

# -----------------------------
# Auxiliar: Top N días UVI
# -----------------------------
def compute_top_days(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["uv_index_max"] = pd.to_numeric(df["uv_index_max"], errors="coerce")
    df = df.dropna(subset=["uv_index_max"])
    return df.nlargest(n, "uv_index_max")[["date", "uv_index_max"]]

# =============================
# COBRE (mindicador.cl)
# =============================
MINDICADOR_BASE = "https://mindicador.cl/api"

@st.cache_data(show_spinner=False)
def fetch_mindicador_series(indicador: str) -> tuple[pd.DataFrame, str | None]:
    """
    Obtiene la serie reciente de un indicador desde mindicador.cl
    Devuelve DataFrame con columnas: date, value; y posible error.
    """
    url = f"{MINDICADOR_BASE}/{indicador}"
    data, err = _safe_json_get(url, None, timeout=60)
    if err:
        return pd.DataFrame(), f"Error {indicador}: {err}"
    serie = data.get("serie", [])
    if not isinstance(serie, list) or not serie:
        return pd.DataFrame(), f"Sin serie para {indicador}"
    df = pd.DataFrame({
        "date": pd.to_datetime([s.get("fecha") for s in serie], errors="coerce"),
        "value": pd.to_numeric([s.get("valor") for s in serie], errors="coerce")
    }).dropna()
    df = df.sort_values("date").reset_index(drop=True)
    return df, None

@st.cache_data(show_spinner=False)
def fetch_cobre_usd_and_usdclp() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Trae:
      - Serie de 'libra_cobre' (USD/libra)
      - Serie de 'dolar' (CLP/USD)
    Retorna (df_cobre, df_usd, meta)
    """
    cobre_df, err1 = fetch_mindicador_series("libra_cobre")
    usd_df, err2 = fetch_mindicador_series("dolar")
    meta = {"error": None}
    if err1 or err2:
        meta["error"] = "; ".join([e for e in [err1, err2] if e])
    return cobre_df, usd_df, meta

def last_value(df: pd.DataFrame) -> tuple[datetime | None, float | None]:
    if df is None or df.empty:
        return None, None
    row = df.iloc[-1]
    return (row["date"].to_pydatetime() if isinstance(row["date"], pd.Timestamp) else None,
            float(row["value"]))

def clip_by_date(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    return df.loc[mask].copy()

# =============================
# UI
# =============================
st.title("☀️ Radiación UV – Norte Grande de Chile")
st.caption("Índice UV histórico (con fallback) y pronóstico basado en Open-Meteo. • Incluye panel de Cobre (libra).")

# ----- Sidebar -----
with st.sidebar:
    st.header("Parámetros UVI")
    ciudad = st.selectbox("Ciudad", list(NORTE_GRANDE_CITIES.keys()), index=3)
    inicio = st.date_input("Desde", DEFAULT_START)
    fin = st.date_input("Hasta", DEFAULT_END)
    dias = st.slider("Pronóstico (días)", 1, 7, 5)

    st.markdown("---")
    st.header("Parámetros Cobre")
    rango_cobre = st.slider("Rango histórico cobre (días)", 30, 180, 90)

# ----- Validación robusta de fechas -----
hoy = date.today()
ayer = AYER
if fin > ayer:
    fin = ayer
if inicio >= fin:
    inicio = max(fin - timedelta(days=30), date(2000, 1, 1))  # 30 días por defecto si está invertido

st.caption(f"⏳ Consultando histórico **{inicio} → {fin}** en **{ciudad}** (hasta ayer: {ayer}).")

# ----- Descarga UVI -----
lat, lon = NORTE_GRANDE_CITIES[ciudad]
hist, err1, dbg = fetch_uv_daily_smart(lat, lon, inicio, fin)
pron, err2 = fetch_uv_forecast_hourly(lat, lon, dias)
st.caption(f"🛰️ Fuente de histórico usada: **{dbg.get('source')}**")

# =============================
# Histórico UVI
# =============================
st.subheader(f"📈 Diario Histórico UVI – {ciudad}")
if err1:
    st.error(err1)
elif hist.empty:
    st.warning("⚠️ Sin datos históricos en este rango. Prueba ampliar el período o cambiar de ciudad.")
else:
    chart_hist = (
        alt.Chart(hist)
        .mark_line(color="orange", point=True)
        .encode(
            x=alt.X("date:T", title="Fecha"),
            y=alt.Y("uv_index_max:Q", title="Índice UV máx"),
            tooltip=["date:T", alt.Tooltip("uv_index_max:Q", title="UVI máx")]
        )
        .properties(height=320, title=f"Histórico UVI – {ciudad}")
    )
    st.altair_chart(chart_hist, use_container_width=True)

    st.markdown("**🌟 Top 5 días con mayor UVI**")
    top5 = compute_top_days(hist, 5)
    if top5.empty:
        st.info("No hay suficientes datos para calcular el Top 5.")
    else:
        top5 = top5.copy()
        top5["date"] = top5["date"].dt.date
        st.table(top5.rename(columns={"date": "Fecha", "uv_index_max": "UVI máx"}))

# =============================
# Pronóstico UVI
# =============================
st.subheader(f"🔮 Pronóstico UVI horario – Próximos {dias} días ({ciudad})")
if err2:
    st.error(err2)
elif pron.empty:
    st.warning("⚠️ Sin datos de pronóstico en este momento.")
else:
    chart_pron = (
        alt.Chart(pron)
        .mark_line(color="red")
        .encode(
            x=alt.X("time:T", title="Hora"),
            y=alt.Y("uv_index:Q", title="Índice UV"),
            tooltip=["time:T", alt.Tooltip("uv_index:Q", title="UVI")]
        )
        .properties(height=320, title=f"Pronóstico UVI – {ciudad}")
    )
    st.altair_chart(chart_pron, use_container_width=True)
    st.dataframe(pron, use_container_width=True, height=260)

# =============================
# Panel de COBRE (mindicador.cl)
# =============================
st.markdown("---")
st.header("🪙 Cobre – Precio por libra")
st.caption("Fuente: API pública de mindicador.cl (indicadores: libra_cobre en USD/libra y dólar observado en CLP/USD).")

cobre_df, usd_df, meta = fetch_cobre_usd_and_usdclp()
if meta.get("error"):
    st.error(meta["error"])
elif cobre_df.empty or usd_df.empty:
    st.warning("No fue posible recuperar la serie de cobre y/o dólar en este momento.")
else:
    # KPIs
    last_date_usd, last_cobre_usd = last_value(cobre_df)
    _, last_usdclp = last_value(usd_df)

    col1, col2, col3 = st.columns([1.2, 1.2, 1])
    with col1:
        st.metric("Cobre (USD/libra)", 
                  value="—" if last_cobre_usd is None else f"{last_cobre_usd:,.4f}",
                  delta=None)
    with col2:
        if last_cobre_usd is not None and last_usdclp is not None:
            cobre_clp = last_cobre_usd * last_usdclp
            st.metric("Cobre (CLP/libra)", f"{cobre_clp:,.0f}")
        else:
            st.metric("Cobre (CLP/libra)", "—")
    with col3:
        if last_usdclp is not None:
            st.metric("USDCLP", f"{last_usdclp:,.0f}")
        else:
            st.metric("USDCLP", "—")
    st.caption(f"Última actualización: {last_date_usd.strftime('%d-%m-%Y') if last_date_usd else '—'}")

    # Rango histórico reciente (por días)
    start_cobre = (datetime.now(timezone.utc) - timedelta(days=int(rango_cobre))).date()
    end_cobre = date.today()
    cobre_clip = clip_by_date(cobre_df, start_cobre, end_cobre)

    if cobre_clip.empty:
        st.info("No hay datos en el rango seleccionado. Prueba con un rango mayor.")
    else:
        # Serie USD/libra
        chart_cobre_usd = (
            alt.Chart(cobre_clip.rename(columns={"value": "usd_lb"}))
            .mark_line(color="#2E86DE")
            .encode(
                x=alt.X("date:T", title="Fecha"),
                y=alt.Y("usd_lb:Q", title="USD/libra"),
                tooltip=["date:T", alt.Tooltip("usd_lb:Q", title="USD/libra", format=".4f")],
            )
            .properties(height=320, title=f"Cobre USD/libra – últimos {rango_cobre} días")
        )

        # Serie CLP/libra derivada (usando último USDCLP del endpoint)
        if last_usdclp:
            cobre_clip_clp = cobre_clip.copy()
            cobre_clip_clp["clp_lb"] = cobre_clip_clp["value"] * last_usdclp
            chart_cobre_clp = (
                alt.Chart(cobre_clip_clp)
                .mark_line(color="#12B886")
                .encode(
                    x=alt.X("date:T", title="Fecha"),
                    y=alt.Y("clp_lb:Q", title="CLP/libra"),
                    tooltip=["date:T", alt.Tooltip("clp_lb:Q", title="CLP/libra", format=",.0f")],
                )
                .properties(height=320, title=f"Cobre CLP/libra (estimado) – últimos {rango_cobre} días")
            )

            tabs = st.tabs(["USD/libra", "CLP/libra (estimado)"])
            with tabs[0]:
                st.altair_chart(chart_cobre_usd, use_container_width=True)
            with tabs[1]:
                st.altair_chart(chart_cobre_clp, use_container_width=True)
        else:
            st.altair_chart(chart_cobre_usd, use_container_width=True)
            st.info("No se pudo calcular CLP/libra porque USDCLP no estuvo disponible.")
