"""
Portfolio Manager & Value Screener (Streamlit)

Ejecutar:  streamlit run portfolio_screener.py

Las URLs de Google Sheets NO van en el código. Definilas en
.streamlit/secrets.toml (o como variables de entorno):

    URL_CARTERA_CSV   = "https://docs.google.com/.../pub?gid=0&single=true&output=csv"
    URL_WATCHLIST_CSV = "https://docs.google.com/.../pub?gid=...&single=true&output=csv"
    URL_RESULTADO_CSV = "https://docs.google.com/.../pub?gid=...&single=true&output=csv"
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ------------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Manager & Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------------------
# CONFIGURACIÓN (URLs fuera del código)
# ------------------------------------------------------------------------------
def _get_config(nombre: str) -> str | None:
    try:
        valor = st.secrets.get(nombre)
        if valor:
            return str(valor)
    except Exception:
        pass  # no hay secrets.toml
    return os.environ.get(nombre)


URLS = {
    "cartera": _get_config("URL_CARTERA_CSV"),
    "watchlist": _get_config("URL_WATCHLIST_CSV"),
    "resultado": _get_config("URL_RESULTADO_CSV"),
}
faltan = [k for k, v in URLS.items() if not v]
if faltan:
    st.error(
        "Faltan las URLs de Google Sheets: "
        + ", ".join(f"URL_{k.upper()}_CSV" for k in faltan)
        + ". Definilas en .streamlit/secrets.toml o como variables de entorno."
    )
    st.stop()

COLS_BASE = ["Ticker", "Origen", "Broker", "Cantidad", "Importe usd", "Cat", "Especie"]
ALIAS_TICKERS = {"BRK.B": "BRK-B", "BF.B": "BF-B"}  # Yahoo usa guion en clases de acciones


# ------------------------------------------------------------------------------
# UTILIDADES DE LIMPIEZA
# ------------------------------------------------------------------------------
def parse_num(val) -> float:
    """Convierte textos tipo '$1.234,56' o '1,5' a float. Coma sola = decimal."""
    if val is None or (not isinstance(val, str) and pd.isna(val)):
        return 0.0
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip().replace("$", "").replace(" ", "")
    if not s or s.lower() == "nan" or s == "-":
        return 0.0
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def normalizar_ticker(raw) -> str:
    if raw is None or pd.isna(raw):
        return ""
    t = str(raw).strip().upper()
    if t in ("", "NAN", "NONE"):
        return ""
    return ALIAS_TICKERS.get(t, t)


def _num(x) -> float | None:
    """float válido o None (descarta None, NaN, inf, textos)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def mostrar_df(df: pd.DataFrame) -> None:
    """Compatible con versiones nuevas y viejas de Streamlit."""
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# INDICADORES TÉCNICOS
# ------------------------------------------------------------------------------
def rsi_serie(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder (suavizado exponencial), el mismo que usan TradingView y similares."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)                       # solo subas -> 100
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)  # precio plano -> 50
    return rsi


# ------------------------------------------------------------------------------
# DESCARGA DE DATOS (yfinance)
# ------------------------------------------------------------------------------
def descargar_historial(ticker: str, period: str = "1y"):
    """Devuelve (símbolo_usado, historial). Prueba sufijo .BA solo si el original falla."""
    candidatos = [ticker]
    if "." not in ticker and "-" not in ticker and len(ticker) <= 5:
        candidatos.append(f"{ticker}.BA")
    for symbol in candidatos:
        try:
            hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        except Exception:
            continue
        if not hist.empty and hist["Close"].dropna().size > 0:
            return symbol, hist
    return None, None


# ------------------------------------------------------------------------------
# SCORING (separado en Fundamental y Técnico)
# ------------------------------------------------------------------------------
def _score(componentes: list[tuple[float | None, float]]) -> float:
    """Promedio ponderado (0-100) solo con los componentes que tienen dato."""
    validos = [(v, w) for v, w in componentes if v is not None and not np.isnan(v)]
    if not validos:
        return np.nan
    total_w = sum(w for _, w in validos)
    return round(100 * sum(v * w for v, w in validos) / total_w, 1)


def score_fundamental(pe, fpe, crec_ing, margen_op) -> float:
    positivos = [x for x in (fpe, pe) if x is not None and x > 0]
    if positivos:
        v_pe = _clip01((35 - positivos[0]) / 30)  # P/E 5 -> 1.0 ; P/E 35+ -> 0.0
    elif any(x is not None for x in (fpe, pe)):
        v_pe = 0.0                                 # ganancias negativas
    else:
        v_pe = None
    v_crec = None if crec_ing is None else _clip01(crec_ing / 0.25)    # 25%+ -> 1.0
    v_marg = None if margen_op is None else _clip01(margen_op / 0.35)  # 35%+ -> 1.0
    return _score([(v_pe, 0.4), (v_crec, 0.3), (v_marg, 0.3)])


def score_tecnico(precio, sma50, sma200, rsi) -> float:
    v200 = None if sma200 is None else float(precio > sma200)
    v50 = None if sma50 is None else float(precio > sma50)
    v_rsi = None if rsi is None else _clip01((70 - rsi) / 40)  # RSI 30 -> 1.0 ; 70 -> 0.0
    return _score([(v200, 0.4), (v50, 0.2), (v_rsi, 0.4)])


def clasificar_tier(score: float) -> str:
    if score is None or np.isnan(score):
        return "Sin datos"
    return "Tier 1" if score > 60 else ("Tier 2" if score > 40 else "Tier 3")


def analizar(ticker: str, period: str = "1y"):
    """Calcula todas las métricas de un ticker. Devuelve (fila_dict | None, historial | None)."""
    symbol, hist = descargar_historial(ticker, period)
    if hist is None:
        return None, None

    close = hist["Close"].dropna()
    precio = float(close.iloc[-1])
    sma50 = _num(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = _num(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    rsi = _num(rsi_serie(close).iloc[-1])
    ventana = close.tail(60)

    # .info falla seguido (rate limit): si falla, igual devolvemos la parte técnica
    info: dict = {}
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        pass
    pe = _num(info.get("trailingPE"))
    fpe = _num(info.get("forwardPE"))
    crec = _num(info.get("revenueGrowth"))
    margen = _num(info.get("operatingMargins"))

    s_fund = score_fundamental(pe, fpe, crec, margen)
    s_tec = score_tecnico(precio, sma50, sma200, rsi)
    s_total = _score([(None if np.isnan(s_fund) else s_fund / 100, 0.5),
                      (None if np.isnan(s_tec) else s_tec / 100, 0.5)])

    fila = {
        "Ticker": ticker,
        "Símbolo Yahoo": symbol,
        "Moneda": info.get("currency"),
        "Precio": round(precio, 2),
        "P/E": None if pe is None else round(pe, 2),
        "P/E fwd": None if fpe is None else round(fpe, 2),
        "Crec. ingresos %": None if crec is None else round(crec * 100, 1),
        "Margen op. %": None if margen is None else round(margen * 100, 1),
        "RSI": None if rsi is None else round(rsi, 1),
        "vs SMA200 %": None if sma200 is None else round((precio / sma200 - 1) * 100, 1),
        "Soporte 60d": round(float(ventana.min()), 2),
        "Resistencia 60d": round(float(ventana.max()), 2),
        "Score_Fund": s_fund,
        "Score_Tec": s_tec,
        "Score": s_total,
        "Tier": clasificar_tier(s_total),
    }
    return fila, hist


def procesar_lote(tickers: list[str], progress_cb) -> list[tuple[str, dict | None]]:
    """Descarga en paralelo (la UI se actualiza solo desde el hilo principal)."""
    resultados: list[tuple[str, dict | None]] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futuros = {ex.submit(analizar, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futuros), start=1):
            t = futuros[fut]
            try:
                fila, _ = fut.result()
            except Exception:
                fila = None
            resultados.append((t, fila))
            progress_cb(i, len(tickers), t)
    return resultados


# ------------------------------------------------------------------------------
# CARGA DE GOOGLE SHEETS (con caché; si falla, NO se cachea el error)
# ------------------------------------------------------------------------------
def clasificar_mercado(cat, ticker: str) -> str:
    cat = "" if cat is None or pd.isna(cat) else str(cat).upper()
    if "USA" in cat or "CEDEAR" in cat or "." not in ticker:
        return "Acciones US / Internacionales"
    return "Resto / Local"


def _preparar(df: pd.DataFrame, origen: str) -> pd.DataFrame:
    if "Ticker" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "Origen"])
    df = df.copy()
    df["Ticker"] = df["Ticker"].map(normalizar_ticker)
    df = df[df["Ticker"] != ""]
    for c in ("Cantidad", "Importe usd"):
        if c in df.columns:
            df[c] = df[c].map(parse_num)
    df["Origen"] = origen
    return df[[c for c in COLS_BASE if c in df.columns]]


@st.cache_data(ttl=300, show_spinner=False)
def cargar_datos_sheets() -> pd.DataFrame:
    df_cart = _preparar(pd.read_csv(URLS["cartera"]), "CARTERA")
    df_watch = _preparar(pd.read_csv(URLS["watchlist"]), "WATCHLIST")
    df_total = pd.concat([df_cart, df_watch], ignore_index=True, sort=False)
    # CARTERA va primero: si un ticker está en ambas hojas, gana CARTERA
    df_total = df_total.drop_duplicates(subset=["Ticker"], keep="first")
    if "Cat" not in df_total.columns:
        df_total["Cat"] = ""
    df_total["Mercado"] = [clasificar_mercado(c, t) for c, t in zip(df_total["Cat"], df_total["Ticker"])]
    return df_total.reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def cargar_resultado() -> pd.DataFrame:
    return pd.read_csv(URLS["resultado"])


# ------------------------------------------------------------------------------
# GRÁFICO
# ------------------------------------------------------------------------------
def graficar(ticker: str, hist: pd.DataFrame) -> None:
    close = hist["Close"].dropna()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(close.index, close, label="Cierre", color="#38bdf8", linewidth=1.5)
    ax1.plot(close.index, close.rolling(50).mean(), label="SMA 50", color="#f59e0b", linewidth=1)
    ax1.plot(close.index, close.rolling(200).mean(), label="SMA 200", color="#ef4444", linewidth=1)
    ventana = close.tail(60)
    ax1.axhline(ventana.min(), color="#22c55e", linestyle="--", alpha=0.6, label="Soporte 60d")
    ax1.axhline(ventana.max(), color="#a855f7", linestyle="--", alpha=0.6, label="Resistencia 60d")
    ax1.set_title(f"{ticker} — Precio y medias móviles")
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    ax2.plot(close.index, rsi_serie(close), color="#38bdf8", linewidth=1)
    ax2.axhline(70, color="#ef4444", linestyle="--", alpha=0.6)
    ax2.axhline(30, color="#22c55e", linestyle="--", alpha=0.6)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI (14)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # evita fuga de memoria en cada rerun


# ------------------------------------------------------------------------------
# INTERFAZ PRINCIPAL
# ------------------------------------------------------------------------------
st.title("📈 Portfolio Manager & Value Screener Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📊 Screener & Filtros", "💼 Mi Cartera", "🔍 Análisis por Ticker"]
)

# ------------------------------- TAB 1 ----------------------------------------
with tab1:
    st.subheader("Screener Value & System Score")

    if st.button("🔄 Cargar / Actualizar Datos del Mercado"):
        try:
            df_base = cargar_datos_sheets()
        except Exception as e:
            st.error(f"Error cargando Google Sheets: {e}")
            df_base = pd.DataFrame()

        if not df_base.empty:
            tickers = df_base["Ticker"].tolist()
            bar = st.progress(0.0, text="Procesando tickers...")

            def _cb(i: int, n: int, t: str) -> None:
                bar.progress(i / n, text=f"Evaluado {t} ({i}/{n})")

            resultados = procesar_lote(tickers, _cb)
            bar.empty()

            filas = [fila for _, fila in resultados if fila]
            fallidos = sorted(t for t, fila in resultados if not fila)

            if filas:
                df_res = pd.DataFrame(filas).merge(
                    df_base[["Ticker", "Origen", "Mercado"]], on="Ticker", how="left"
                )
                st.session_state["df_screener"] = df_res
                st.success(f"Métricas calculadas: {len(filas)} de {len(tickers)} tickers.")
            else:
                st.warning("No se pudieron obtener métricas de Yahoo para ningún ticker.")
            if fallidos:
                st.warning("Sin datos en Yahoo (¿bonos o falta sufijo?): " + ", ".join(fallidos))

    if "df_screener" in st.session_state:
        df = st.session_state["df_screener"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            f_origen = st.multiselect("Origen:", df["Origen"].dropna().unique(),
                                      default=df["Origen"].dropna().unique())
        with c2:
            f_mercado = st.multiselect("Mercado:", df["Mercado"].dropna().unique(),
                                       default=df["Mercado"].dropna().unique())
        with c3:
            f_tier = st.multiselect("Tier:", df["Tier"].unique(), default=df["Tier"].unique())
        with c4:
            buscar = st.text_input("Buscar Ticker:", "").strip().upper()

        df_f = df[df["Origen"].isin(f_origen) & df["Mercado"].isin(f_mercado) & df["Tier"].isin(f_tier)]
        if buscar:
            df_f = df_f[df_f["Ticker"].str.contains(buscar, regex=False)]

        mostrar_df(df_f.sort_values("Score", ascending=False, na_position="last"))

        if (df_f["Moneda"].dropna().nunique() > 1) or df_f["Símbolo Yahoo"].str.endswith(".BA").any():
            st.caption("⚠️ Hay tickers en distintas monedas (ej. .BA = pesos). "
                       "Los precios no son comparables entre sí.")
        st.caption("Score = 50% fundamental (P/E, crecimiento de ingresos, margen operativo) + "
                   "50% técnico (SMA50/200, RSI). Los datos faltantes se excluyen del promedio, no suman ni restan.")
    else:
        st.info("Presioná el botón de arriba para consultar las métricas en vivo de tus listas.")

# ------------------------------- TAB 2 ----------------------------------------
with tab2:
    st.subheader("Resumen de Posiciones y Resultados")
    try:
        mostrar_df(cargar_resultado())
    except Exception as e:
        st.error(f"Error al cargar la hoja de resultados: {e}")

# ------------------------------- TAB 3 ----------------------------------------
with tab3:
    st.subheader("Generador de Dashboard por Ticker")
    ticker_input = normalizar_ticker(st.text_input("Ingresá un Ticker para analizar:", value="AAPL"))

    if st.button("📊 Generar Análisis") and ticker_input:
        with st.spinner(f"Buscando datos para {ticker_input}..."):
            fila, hist = analizar(ticker_input, period="2y")
        if fila is None:
            st.session_state.pop("analisis", None)
            st.warning("No se encontraron datos históricos para este ticker en Yahoo Finance.")
        else:
            st.session_state["analisis"] = {"ticker": ticker_input, "fila": fila, "hist": hist}

    # Se guarda en session_state para que no desaparezca al tocar otro widget
    if "analisis" in st.session_state:
        a = st.session_state["analisis"]
        fila = a["fila"]
        if fila["Símbolo Yahoo"] != a["ticker"]:
            st.warning(f"No se encontró {a['ticker']}; se usó {fila['Símbolo Yahoo']} "
                       "(precio probablemente en pesos).")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Precio", fila["Precio"])
        m2.metric("RSI (14)", fila["RSI"] if fila["RSI"] is not None else "s/d")
        m3.metric("vs SMA200", f"{fila['vs SMA200 %']}%" if fila["vs SMA200 %"] is not None else "s/d")
        m4.metric("P/E", fila["P/E"] if fila["P/E"] is not None else "s/d")
        m5.metric("Tier", fila["Tier"])
        graficar(a["ticker"], a["hist"])
        mostrar_df(pd.DataFrame([fila]))
