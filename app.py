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
# FUNCIONES AUXILIARES DE LIMPIEZA
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
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# ------------------------------------------------------------------------------
# CARGA DE DATOS (CON CACHÉ PARA MAYOR VELOCIDAD)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def cargar_datos_sheets():
    try:
        df_cart = pd.read_csv(URL_CARTERA_CSV)
        df_watch = pd.read_csv(URL_WATCHLIST_CSV)
        
        df_cart = df_cart[df_cart["Ticker"].astype(str).str.strip() != ""].copy()
        df_cart["Origen"] = "CARTERA"
        
        df_watch = df_watch[df_watch["Ticker"].astype(str).str.strip() != ""].copy()
        df_watch["Origen"] = "WATCHLIST"
        
        df_total = pd.concat([df_cart[["Ticker", "Origen"]], df_watch[["Ticker", "Origen"]]], ignore_index=True)
        df_total = df_total.drop_duplicates(subset=["Ticker"], keep="first")
        return df_total
    except Exception as e:
        st.error(f"Error cargando Google Sheets: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def calcular_metricas_ticker(ticker_sym):
    ticker_yf = ticker_sym.replace(".", "-")
    try:
        t = yf.Ticker(ticker_yf)
        hist = t.history(period="3y")
        if len(hist) < 200:
            return None

        hist["EMA200"] = hist["Close"].ewm(span=200, adjust=False).mean()
        precio_actual = hist["Close"].iloc[-1]
        ema200_actual = hist["EMA200"].iloc[-1]

        dist_ema200_pct = ((precio_actual - ema200_actual) / ema200_actual) * 100
        rsi = calcular_rsi(hist["Close"], 14)

        hist["Dist_EMA200_Hist"] = ((hist["Close"] - hist["EMA200"]) / hist["EMA200"]) * 100
        desvio_pos_prom = hist[hist["Dist_EMA200_Hist"] > 0]["Dist_EMA200_Hist"].mean()
        desvio_neg_prom = hist[hist["Dist_EMA200_Hist"] < 0]["Dist_EMA200_Hist"].mean()

        low_52 = hist["Low"].tail(252).min()
        high_52 = hist["High"].tail(252).max()
        pos_52_pct = ((precio_actual - low_52) / (high_52 - low_52)) * 100 if high_52 > low_52 else 50.0

        info = t.info
        market_cap = info.get("marketCap", 0)
        free_cashflow = info.get("freeCashflow", None)
        fcf_yield = (free_cashflow / market_cap * 100) if (free_cashflow and market_cap) else None

        peg_ratio = info.get("pegRatio", None)
        roe = info.get("returnOnEquity", None)
        roe_pct = roe * 100 if roe else None

        debt_to_equity = info.get("debtToEquity", None)
        current_ratio = info.get("currentRatio", None)

        revenue_growth = info.get("revenueGrowth", None)
        rev_growth_pct = revenue_growth * 100 if revenue_growth else None

        earnings_growth = info.get("earningsGrowth", None)
        earn_growth_pct = earnings_growth * 100 if earnings_growth else None

        profit_margins = info.get("profitMargins", None)
        target_price = info.get("targetMeanPrice", None)

        upside_target_pct = None
        if target_price and precio_actual > 0:
            upside_target_pct = ((target_price - precio_actual) / precio_actual) * 100

        # CÁLCULO DEL SCORE
        score_fund, score_tec, score_riesgo = 0, 0, 0

        if fcf_yield is not None:
            if fcf_yield > 5.0: score_fund += 2
            elif 3.0 <= fcf_yield <= 5.0: score_fund += 1

        if roe_pct is not None and debt_to_equity is not None:
            if roe_pct > 15.0 and debt_to_equity < 100.0: score_fund += 2
            elif debt_to_equity > 200.0: score_fund -= 2

        if current_ratio and current_ratio > 1.5: score_fund += 1

        if rev_growth_pct is not None and earn_growth_pct is not None:
            if rev_growth_pct > 10.0 and earn_growth_pct > 10.0: score_fund += 2
            elif rev_growth_pct > 10.0 or earn_growth_pct > 10.0: score_fund += 1
            elif rev_growth_pct < 0.0 and earn_growth_pct < 0.0: score_fund -= 2

        if pd.notna(desvio_neg_prom) and dist_ema200_pct <= desvio_neg_prom and rsi < 45:
            score_tec += 2
        if rsi > 75:
            score_tec -= 1

        if rev_growth_pct is not None and rev_growth_pct < 0.0 and (profit_margins and profit_margins < 0.05):
            score_riesgo -= 3

        score_total = score_fund + score_tec + score_riesgo

        if score_total >= 6: tier = "🟢 PRIORITARIO"
        elif 2 <= score_total <= 5: tier = "🟡 VIGILAR"
        elif -1 <= score_total <= 1: tier = "⚪ NEUTRAL"
        else: tier = "🔴 TRAMPA"

        return {
            "Ticker": ticker_sym,
            "Precio": round(precio_actual, 2),
            "Score": score_total,
            "Tier": tier,
            "FCF Yield %": round(fcf_yield, 2) if fcf_yield is not None else None,
            "Dist EMA200 %": round(dist_ema200_pct, 2),
            "RSI": round(rsi, 2) if pd.notna(rsi) else None,
            "ROE %": round(roe_pct, 2) if roe_pct else None,
            "D/E": round(debt_to_equity, 1) if debt_to_equity else None,
            "Crec. Ventas %": round(rev_growth_pct, 2) if rev_growth_pct else None,
            "Upside %": round(upside_target_pct, 2) if upside_target_pct else None,
        }
    except Exception:
        return None

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
            
            df_res = pd.DataFrame(resultados)
            st.session_state["df_screener"] = df_res

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
        # Aquí Streamlit mostrará el gráfico de Matplotlib directamente
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
