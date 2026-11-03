import os
import io
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA (STREAMLIT)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Manager & Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# ENLACES GOOGLE SHEETS
# ------------------------------------------------------------------------------
URL_CARTERA_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRbLGRdor-TtNOtkqL0cbrTnUN0mg6-FLM-3yAxuZsznZRUJjeqoyWC7ZubG6kp1SEgYvcryTnb1eyE/pub?gid=0&single=true&output=csv"
URL_WATCHLIST_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRbLGRdor-TtNOtkqL0cbrTnUN0mg6-FLM-3yAxuZsznZRUJjeqoyWC7ZubG6kp1SEgYvcryTnb1eyE/pub?gid=440350475&single=true&output=csv"
URL_RESULTADO_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRbLGRdor-TtNOtkqL0cbrTnUN0mg6-FLM-3yAxuZsznZRUJjeqoyWC7ZubG6kp1SEgYvcryTnb1eyE/pub?gid=264036849&single=true&output=csv"

# ------------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE LIMPIEZA Y MÉTRICAS
# ------------------------------------------------------------------------------
def parse_num(val):
    if pd.isna(val) or val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip().replace("$", "").replace(" ", "")
    if not val_str or val_str.lower() == "nan" or val_str == "-":
        return 0.0
    if "," in val_str and "." in val_str:
        val_str = val_str.replace(".", "").replace(",", ".")
    elif "," in val_str:
        val_str = val_str.replace(",", ".")
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def parse_pct(val):
    if pd.isna(val) or val is None:
        return 0.0
    val_str = str(val).strip()
    tiene_signo_pct = "%" in val_str
    num_val = parse_num(val_str.replace("%", ""))
    if tiene_signo_pct:
        return num_val
    if -1.0 <= num_val <= 1.0 and num_val != 0:
        return num_val * 100.0
    return num_val

def calcular_rsi(series, period=14):
    if len(series) < period:
        return 0.0
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]):
        return float(rsi.iloc[-1])
    return 0.0

def calcular_metricas_ticker(ticker):
    """Calcula métricas básicas de mercado y un score de ejemplo usando yfinance"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="6mo")
        if hist.empty:
            return None
        
        precio_actual = hist["Close"].iloc[-1]
        rsi_val = calcular_rsi(hist["Close"])
        
        # Obtener info fundamental básica de manera segura
        info = t.info
        pe = info.get("trailingPE", 0.0)
        if pe is None: 
            pe = 0.0
            
        # Simulación de un Score y Tier básico para que el screener no falle
        score = max(0, min(100, 50 - (pe if pe > 0 else 20) + (100 - rsi_val) * 0.2))
        tier = "Tier 1" if score > 60 else ("Tier 2" if score > 40 else "Tier 3")
        
        return {
            "Ticker": ticker,
            "Precio": round(precio_actual, 2),
            "P/E": round(float(pe), 2),
            "RSI": round(rsi_val, 2),
            "Score": round(score, 2),
            "Tier": tier
        }
    except Exception:
        return None

# ------------------------------------------------------------------------------
# CARGA DE DATOS (CON CACHÉ Y PROTECCIÓN CONTRA KEYERROR)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def cargar_datos_sheets():
    try:
        df_cart = pd.read_csv(URL_CARTERA_CSV)
        df_watch = pd.read_csv(URL_WATCHLIST_CSV)
        
        # Validar y limpiar Cartera de forma segura
        if "Ticker" in df_cart.columns:
            df_cart = df_cart[df_cart["Ticker"].astype(str).str.strip() != ""].copy()
            df_cart["Origen"] = "CARTERA"
        else:
            df_cart["Origen"] = "CARTERA"
            df_cart["Ticker"] = ""

        # Validar y limpiar Watchlist de forma segura
        if "Ticker" in df_watch.columns:
            df_watch = df_watch[df_watch["Ticker"].astype(str).str.strip() != ""].copy()
            df_watch["Origen"] = "WATCHLIST"
        else:
            df_watch["Origen"] = "WATCHLIST"
            df_watch["Ticker"] = ""
        
        # Filtrar columnas existentes para evitar KeyError al concatenar
        cols_cart = [c for c in ["Ticker", "Origen", "Broker", "Cantidad", "Importe usd"] if c in df_cart.columns]
        cols_watch = [c for c in ["Ticker", "Origen", "Broker", "Cantidad", "Importe usd"] if c in df_watch.columns]
        
        df_total = pd.concat([df_cart[cols_cart], df_watch[cols_watch]], ignore_index=True, sort=False)
        df_total = df_total.drop_duplicates(subset=["Ticker"], keep="first")
        return df_total
        
    except Exception as e:
        st.error(f"Error cargando Google Sheets: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# INTERFAZ PRINCIPAL
# ------------------------------------------------------------------------------
st.title("📈 Portfolio Manager & Value Screener Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Screener & Filtros", "💼 Mi Cartera", "🔍 Análisis por Ticker (/pe)"])

# ------------------------------------------------------------------------------
# TAB 1: SCREENER CON FILTROS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Screener Value & System Score")
    
    if st.button("🔄 Cargar / Actualizar Datos del Mercado"):
        df_base = cargar_datos_sheets()
        if not df_base.empty:
            resultados = []
            bar = st.progress(0, text="Procesando tickers...")
            tickers = df_base["Ticker"].tolist()
            for idx, t in enumerate(tickers):
                m = calcular_metricas_ticker(str(t).strip())
                if m:
                    m["Origen"] = df_base.loc[df_base["Ticker"] == t, "Origen"].values[0]
                    resultados.append(m)
                bar.progress((idx + 1) / len(tickers), text=f"Evaluando {t}...")
            bar.empty()
            
            if resultados:
                df_res = pd.DataFrame(resultados)
                st.session_state["df_screener"] = df_res
                st.success("¡Métricas calculadas con éxito!")
            else:
                st.warning("No se pudieron obtener métricas para los tickers de las listas.")

    if "df_screener" in st.session_state:
        df = st.session_state["df_screener"]

        # Filtros de la Barra Lateral / Arriba
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filtro_origen = st.multiselect("Origen:", options=df["Origen"].unique(), default=df["Origen"].unique())
        with col_f2:
            filtro_tier = st.multiselect("Clasificación (Tier):", options=df["Tier"].unique(), default=df["Tier"].unique())
        with col_f3:
            buscar_ticker = st.text_input("Buscar Ticker:", "").upper()

        # Filtrar DataFrame
        df_filtrado = df[
            (df["Origen"].isin(filtro_origen)) &
            (df["Tier"].isin(filtro_tier))
        ]
        if buscar_ticker:
            df_filtrado = df_filtrado[df_filtrado["Ticker"].str.contains(buscar_ticker)]

        st.dataframe(
            df_filtrado.sort_values(by="Score", ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Presiona el botón de arriba para consultar las métricas en vivo de tus listas.")

# ------------------------------------------------------------------------------
# TAB 2: MI CARTERA
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Resumen de Posiciones y Resultados")
    try:
        df_res = pd.read_csv(URL_RESULTADO_CSV)
        st.dataframe(df_res, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error al cargar la hoja de resultados: {e}")

# ------------------------------------------------------------------------------
# TAB 3: ANÁLISIS INDIVIDUAL
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Generador de Dashboard por Ticker")
    ticker_input = st.text_input("Ingresa un Ticker para analizar:", value="AAPL").upper()
    
    if st.button("📊 Generar Análisis"):
        st.write(f"Procesando gráfico y checklist fundamental para **{ticker_input}**...")
        t_obj = yf.Ticker(ticker_input)
        hist = t_obj.history(period="2y")
        if not hist.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hist.index, hist["Close"], label="Precio de Cierre", color="#38bdf8")
            ax.set_title(f"Evolución de Precio: {ticker_input}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("No se encontraron datos históricos para este ticker.")
