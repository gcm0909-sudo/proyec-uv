# -*- coding: utf-8 -*-
# Archivo sugerido: pages/02_Radiacion_UV_Norte_Grande.py  (para multipágina)
# o app_uv.py (si lo usas como app única)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
from datetime import date, timedelta, datetime, timezone

st.set_page_config(page_title="Radiación UV – Norte Grande", page_icon="☀️", layout="wide")

# ---------------------------------
# Utilidades y catálogos
# ---------------------------------
NORTE_GRANDE_CITIES = {
    "Arica": (-18.4783, -70.3126),
    "Iquique": (-20.2133, -70.1503),
    "Antofagasta": (-23.6500, -70.4000),
    "Calama": (-22.4550, -68.9290),
    "San Pedro de Atacama": (-22.9110, -68.2030),
    "Tocopilla": (-22.0887, -70.1936),
}

DEFAULT_END = date.today() - timedelta(days=1)
DEFAULT_START = DEFAULT_END - timedelta(days=365)

UVI_BANDS = [
    (0, 2, "Bajo", "#2DC653"),
    (3, 5, "Moderado", "#FFD43B"),
    (6, 7, "Alto", "#FF922B"),
    (8, 10, "Muy alto", "#FA5252"),
    (11, 99, "Extremo", "#862E9C"),
]

def uvi_category(uvi: float):
    if uvi is None or np.isnan(uvi):
        return ("Sin dato", "#adb5bd")
    for lo, hi, label, color in UVI_BANDS:
        if lo <= uvi <= hi:
            return (label, color)
    return ("Sin dato", "#adb5bd")

@st.cache_data(show_spinner=False)
def fetch_uv_archive_daily(lat, lon, start, end):
    """Descarga UVI diario histórico desde Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": "uv_index_max,uv_index_clear_sky_max",
        "timezone": "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return pd.DataFrame(), f"Error al descargar histórico: {e}"

    if "daily" not in data:
        return pd.DataFrame(), "Sin sección 'daily' en la respuesta."

    daily = data["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    df["uv_index_max"] = daily.get("uv_index_max", [np.nan] * len(df))
    df["uv_index_clear_sky_max"] = daily.get("uv_index_clear_sky_max", [np.nan] * len(df))
    df = df.sort_values("date").reset_index(drop=True)
    return df, None

@st.cache_data(show_spinner=False)
def fetch_uv_forecast_hourly(lat, lon, days=7):
    """Descarga UVI horario de pronóstico próximo desde Open-Meteo Forecast API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "uv_index,uv_index_clear_sky",
        "timezone": "auto",
        "forecast_days": int(days),
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return pd.DataFrame(), f"Error al descargar pronóstico: {e}"

    if "hourly" not in data:
        return pd.DataFrame(), "Sin sección 'hourly' en la respuesta."

    h = data["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"])})
    df["uv_index"] = h.get("uv_index", [np.nan] * len(df))
    df["uv_index_clear_sky"] = h.get("uv_index_clear_sky", [np.nan] * len(df))
    df["date"] = df["time"].dt.date
    return df.sort_values("time").reset_index(drop=True), None

def compute_kpis_daily(df):
    if df.empty:
        return {"días": 0, "UVI máx promedio": np.nan, "UVI máx histórico": np.nan, "Fecha UVI pico": None}
    k = {}
    k["días"] = len(df)
    k["UVI máx promedio"] = round(df["uv_index_max"].mean(), 1)
    idxmax = df["uv_index_max"].idxmax()
    if pd.notna(idxmax):
        k["UVI máx histórico"] = round(df.loc[idxmax, "uv_index_max"], 1)
        k["Fecha UVI pico"] = df.loc[idxmax, "date"].date()
    else:
        k["UVI máx histórico"] = np.nan
        k["Fecha UVI pico"] = None
    return k

def monthly_summary_uv(df):
    if df.empty:
        return df
    m = df.copy()
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()
    agg = m.groupby("month").agg(
        UVImax_prom=("uv_index_max", "mean"),
        UVImax_cieloDespejado_prom=("uv_index_clear_sky_max", "mean"),
        UVImax_abs=("uv_index_max", "max"),
    ).reset_index()
    return agg

def download_button_csv(df, filename="uv_norte_grande.csv"):
    if df.empty:
        st.info("No hay datos para descargar.")
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv, file_name=filename, mime="text/csv")

# ---------------------------------
# UI
# ---------------------------------
st.title("☀️ Radiación UV – Norte Grande de Chile")
st.caption("Índice UV histórico y pronóstico para ciudades del Norte Grande. Fuente: Open-Meteo.")

# ----- Sidebar (con límites de fecha válidos) -----
with st.sidebar:
    st.header("Parámetros")
    ayer = date.today() - timedelta(days=1)
    ciudad = st.selectbox("Ciudad", list(NORTE_GRANDE_CITIES.keys()), index=3)
    inicio = st.date_input("Desde", DEFAULT_START, max_value=ayer)
    fin = st.date_input("Hasta", DEFAULT_END, min_value=inicio, max_value=ayer)
    dias = st.slider("Pronóstico (días)", 1, 7, 5)

# ----- Validación de fechas para el histórico -----
hoy = date.today()
ayer = hoy - timedelta(days=1)

ini_sel = inicio
fin_sel = fin

# Recorta futuro
if fin_sel > ayer:
    st.info(f"El histórico solo llega hasta ayer ({ayer}). Se ajustó 'Hasta' a {ayer}.")
    fin_sel = ayer

# Rango mínimo válido
if ini_sel >= fin_sel:
    nuevo_ini = max(fin_sel - timedelta(days=30), fin_sel - timedelta(days=1))
    st.warning(f"El rango no es válido (Desde ≥ Hasta). Se ajustó 'Desde' a {nuevo_ini}.")
    ini_sel = nuevo_ini

# ----- Datos -----
lat, lon = NORTE_GRANDE_CITIES[ciudad]
hist, err1 = fetch_uv_archive_daily(lat, lon, ini_sel, fin_sel)
pron, err2 = fetch_uv_forecast_hourly(lat, lon, dias)

# ----- Histórico diario -----
st.subheader(f"Histórico UVI diario – {ciudad}")
if err1:
    st.error(err1)
else:
    # KPIs
    kpi = compute_kpis_daily(hist)
    c1, c2, c3 = st.columns(3)
    c1.metric("Días", kpi["días"])
    c2.metric("UVI máx prom.", kpi["UVI máx promedio"])
    c3.metric("UVI máx hist.", kpi["UVI máx histórico"])

    # Gráfico
    chart_hist = (
        alt.Chart(hist)
        .mark_line()
        .encode(x="date:T", y="uv_index_max:Q", tooltip=["date:T", "uv_index_max:Q"])
        .properties(height=280)
    )
    st.altair_chart(chart_hist, use_container_width=True)

    # Tabla y descarga
    st.dataframe(hist, use_container_width=True, height=240)
    download_button_csv(hist, f"uv_{ciudad}.csv")

# ----- Pronóstico horario -----
st.subheader(f"Pronóstico UVI horario – Próximos {dias} días")
if err2:
    st.error(err2)
else:
    chart_pron = (
        alt.Chart(pron)
        .mark_line()
        .encode(x="time:T", y="uv_index:Q", tooltip=["time:T", "uv_index:Q"])
        .properties(height=280)
    )
    st.altair_chart(chart_pron, use_container_width=True)
    st.dataframe(pron, use_container_width=True, height=240)
