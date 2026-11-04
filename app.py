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
# URLs DE GOOGLE SHEETS (Directas para asegurar funcionamiento inmediato)
# ------------------------------------------------------------------------------
URL_CARTERA_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRbLGRdor-TtNOtkqL0cbrTnUN0mg6-FLM-3yAxuZsznZRUJjeqoyWC7ZubG6kp1SEgYvcryTnb1eyE/pub?gid=0&single=true&output=csv"
URL_WATCHLIST_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRbLGRdor-TtNOtkqL0cbrTnUN0mg6-FLM-3yAxuZsznZRUJjeqoyWC7ZubG6kp1SEgYvcryTnb1eyE/pub?gid=440350475&single=true&output=csv"
URL_RESULTADO_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRbLGRdor-TtNOtkqL0cbrTnUN0mg6-FLM-3yAxuZsznZRUJjeqoyWC7ZubG6kp1SEgYvcryTnb1eyE/pub?gid=264036849&single=true&output=csv"

COLS_BASE = ["Ticker", "Origen", "Broker", "Cantidad", "Importe usd", "Cat", "Especie"]
ALIAS_TICKERS = {"BRK.B": "BRK-B", "BF.B": "BF-B"}

# ------------------------------------------------------------------------------
# UTILIDADES DE LIMPIEZA
# ------------------------------------------------------------------------------
def parse_num(val) -> float:
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
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None

def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def mostrar_df(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# INDICADORES TÉCNICOS
# ------------------------------------------------------------------------------
def rsi_serie(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi

# ------------------------------------------------------------------------------
# DESCARGA DE DATOS (yfinance)
# ------------------------------------------------------------------------------
def descargar_historial(ticker: str, period: str = "1y"):
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
# SCORING
# ------------------------------------------------------------------------------
def _score(componentes: list[tuple[float | None, float]]) -> float:
    validos = [(v, w) for v, w in componentes if v is not None and not np.isnan(v)]
    if not validos:
        return np.nan
    total_w = sum(w for _, w in validos)
    return round(100 * sum(v * w for v, w in validos) / total_w, 1)

def score_fundamental(pe, fpe, crec_ing, margen_op) -> float:
    positivos = [x for x in (fpe, pe) if x is not None and x > 0]
    if positivos:
        v_pe = _clip01((35 - positivos[0]) / 30)
    elif any(x is not None for x in (fpe, pe)):
        v_pe = 0.0
    else:
        v_pe = None
    v_crec = None if crec_ing is None else _clip01(crec_ing / 0.25)
    v_marg = None if margen_op is None else _clip01(margen_op / 0.35)
    return _score([(v_pe, 0.4), (v_crec, 0.3), (v_marg, 0.3)])

def score_tecnico(precio, sma50, sma200, rsi) -> float:
    v200 = None if sma200 is None else float(precio > sma200)
    v50 = None if sma50 is None else float(precio > sma50)
    v_rsi = None if rsi is None else _clip01((70 - rsi) / 40)
    return _score([(v200, 0.4), (v50, 0.2), (v_rsi, 0.4)])

def clasificar_tier(score: float) -> str:
    if score is None or np.isnan(score):
        return "Sin datos"
    return "Tier 1" if score > 60 else ("Tier 2" if score > 40 else "Tier 3")

def analizar(ticker: str, period: str = "1y"):
    symbol, hist = descargar_historial(ticker, period)
    if hist is None:
        return None, None

    close = hist["Close"].dropna()
    precio = float(close.iloc[-1])
    sma50 = _num(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = _num(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    rsi = _num(rsi_serie(close).iloc[-1])
    ventana = close.tail(60)

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
# CARGA DE GOOGLE SHEETS
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
    df_cart = _preparar(pd.read_csv(URL_CARTERA_CSV), "CARTERA")
    df_watch = _preparar(pd.read_csv(URL_WATCHLIST_CSV), "WATCHLIST")
    df_total = pd.concat([df_cart, df_watch], ignore_index=True, sort=False)
    df_total = df_total.drop_duplicates(subset=["Ticker"], keep="first")
    if "Cat" not in df_total.columns:
        df_total["Cat"] = ""
    df_total["Mercado"] = [clasificar_mercado(c, t) for c, t in zip(df_total["Cat"], df_total["Ticker"])]
    return df_total.reset_index(drop=True)

@st.cache_data(ttl=300, show_spinner=False)
def cargar_resultado() -> pd.DataFrame:
    return pd.read_csv(URL_RESULTADO_CSV)

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
    plt.close(fig)

# ------------------------------------------------------------------------------
# INTERFAZ PRINCIPAL
# ------------------------------------------------------------------------------
st.title("📈 Portfolio Manager & Value Screener Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📊 Screener & Filtros", "💼 Mi Cartera", "🔍 Análisis por Ticker"]
)
# ------------------------------------------------------------------------------
# FUNCIONES DE ANÁLISIS AVANZADO POR TICKER (Estilo Telegram /pe)
# ------------------------------------------------------------------------------
def generar_dashboard_grafico(ticker_str: str, d_bajo: float, d_medio: float, d_alto: float):
    ticker_symbol = ticker_str.strip().upper().replace(".", "-")
    ticker_obj = yf.Ticker(ticker_symbol)
    
    try:
        earnings_data = ticker_obj.get_earnings_dates(limit=50)
    except Exception:
        earnings_data = None

    if earnings_data is None or earnings_data.empty:
        return None, "No se encontraron fechas de ganancias / EPS para este ticker en Yahoo Finance."

    try:
        df_eps = earnings_data[['Reported EPS']].dropna().copy()
        df_eps = df_eps.sort_index()
        df_eps.index = df_eps.index.date
        df_eps.index.name = 'Fecha de Reporte'
        df_eps.columns = ['EPS Reportado']
        df_eps = df_eps.reset_index()
        df_eps["Fecha de Reporte"] = pd.to_datetime(df_eps["Fecha de Reporte"]).dt.normalize()
        df_eps["EPS Reportado"] = df_eps["EPS Reportado"].astype(float)
        df_eps = df_eps.sort_values("Fecha de Reporte")
        df_eps["eps_ttm"] = df_eps["EPS Reportado"].rolling(window=4).sum()

        df_eps["desvio_bajo"] = d_bajo
        df_eps["desvio_medio"] = d_medio
        df_eps["desvio_alto"] = d_alto

        df_eps["Precio_bajo"] = df_eps["desvio_bajo"] * df_eps["eps_ttm"]
        df_eps["Precio_medio"] = df_eps["desvio_medio"] * df_eps["eps_ttm"]
        df_eps["Precio_alto"] = df_eps["desvio_alto"] * df_eps["eps_ttm"]

        benchmark = "SPY"
        start_date = "2020-01-01"
        data_raw = yf.download([ticker_symbol, benchmark], start=start_date, auto_adjust=True, progress=False)

        if isinstance(data_raw.columns, pd.MultiIndex):
            prices = data_raw['Close'][ticker_symbol].to_frame(name='Close')
            spy_prices = data_raw['Close'][benchmark]
        else:
            prices = data_raw[['Close']].copy()
            spy_prices = data_raw['Close']

        df_union = df_eps.set_index("Fecha de Reporte")
        df = prices.join(df_union[['eps_ttm', 'Precio_bajo', 'Precio_medio', 'Precio_alto']])
        cols_interp = ['eps_ttm', 'Precio_bajo', 'Precio_medio', 'Precio_alto']
        df[cols_interp] = df[cols_interp].interpolate(method='linear').ffill().bfill()

        df['RSI'] = rsi_serie(df['Close'])
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['Disp_200'] = ((df['Close'] - df['EMA200']) / df['EMA200']) * 100
        d_avg_pos = df['Disp_200'][df['Disp_200'] > 0].mean()
        d_avg_neg = df['Disp_200'][df['Disp_200'] < 0].mean()
        df['RS'] = df['Close'] / spy_prices
        df['RS_SMA'] = df['RS'].rolling(window=50).mean()

        plt.style.use("dark_background")
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1, figsize=(11, 13), sharex=True,
            gridspec_kw={'height_ratios': [2.5, 1, 1, 1]}, dpi=150
        )

        # Panel 1: Precio y Bandas de P/E
        ax1.plot(df.index, df['Close'], color='#38bdf8', lw=1.5, label=f'Precio {ticker_symbol}')
        ax1.plot(df.index, df['Precio_medio'], color='#facc15', ls='--', alpha=0.8, label=f'P/E Medio ({d_medio:.1f}x)')
        ax1.fill_between(df.index, df['Precio_bajo'], df['Precio_alto'], color='#94a3b8', alpha=0.15, label='Zona de Valor (Bandas P/E)')
        ax1.set_title(f"Dashboard P/E & Multi-Panel: {ticker_symbol}", fontsize=13, fontweight='bold', color='#f8fafc', pad=10)
        ax1.legend(loc='upper left', fontsize='small', facecolor='#1e293b', edgecolor='#334155')
        ax1.grid(True, alpha=0.2)

        # Panel 2: Fuerza Relativa vs SPY
        ax2.plot(df.index, df['RS'], color='#818cf8', label='RS vs SPY')
        ax2.plot(df.index, df['RS_SMA'], color='#fb923c', ls=':', label='RS SMA 50')
        ax2.fill_between(df.index, df['RS'], df['RS_SMA'], where=(df['RS'] > df['RS_SMA']), color='#4ade80', alpha=0.2)
        ax2.set_ylabel("Fuerza Rel.", fontsize=9)
        ax2.legend(loc='upper left', fontsize='x-small', facecolor='#1e293b', edgecolor='#334155')
        ax2.grid(True, alpha=0.2)

        # Panel 3: Dispersión respecto a la EMA 200
        ax3.plot(df.index, df['Disp_200'], color='#c084fc', lw=1)
        ax3.axhline(0, color='#94a3b8', lw=0.8)
        ax3.axhline(d_avg_pos, color='#4ade80', ls='--', alpha=0.5)
        ax3.axhline(d_avg_neg, color='#f87171', ls='--', alpha=0.5)
        ax3.fill_between(df.index, df['Disp_200'], 0, where=(df['Disp_200']>=0), color='#4ade80', alpha=0.1)
        ax3.fill_between(df.index, df['Disp_200'], 0, where=(df['Disp_200']<0), color='#f87171', alpha=0.1)
        ax3.set_ylabel("% Disp. EMA200", fontsize=9)
        ax3.grid(True, alpha=0.2)

        # Panel 4: RSI (14)
        ax4.plot(df.index, df['RSI'], color='#2dd4bf', lw=1)
        ax4.axhline(70, color='#f87171', ls='--', alpha=0.4)
        ax4.axhline(30, color='#4ade80', ls='--', alpha=0.4)
        ax4.set_ylim(0, 100)
        ax4.set_ylabel("RSI (14)", fontsize=9)
        ax4.grid(True, alpha=0.2)

        plt.tight_layout()

        image_stream = io.BytesIO()
        plt.savefig(image_stream, format="png", bbox_inches="tight", facecolor="#0f172a")
        image_stream.seek(0)
        plt.close(fig)

        pe_actual = df['Close'].iloc[-1] / df['eps_ttm'].iloc[-1] if df['eps_ttm'].iloc[-1] > 0 else 0
        rsi_act = df['RSI'].iloc[-1]
        disp_act = df['Disp_200'].iloc[-1]
        rs_lidera = df['RS'].iloc[-1] > df['RS_SMA'].iloc[-1]

        status_txt = {
            "pe": pe_actual,
            "rsi": rsi_act,
            "disp": disp_act,
            "d_neg": d_avg_neg,
            "rs_lidera": rs_lidera
        }

        return image_stream, status_txt
    except Exception as e:
        return None, f"Error procesando datos históricos/EPS: {str(e)}"

def check_pm_pro(ticker_str: str):
    try:
        asset = yf.Ticker(ticker_str)
        info = asset.info
        fin = asset.financials.T
        cf = asset.cashflow.T

        price = info.get('currentPrice') or info.get('regularMarketPrice') or 1
        target = info.get('targetMeanPrice', price)
        upside = ((target - price) / price) * 100 if target else 0

        rev_now = fin['Total Revenue'].iloc[0] if 'Total Revenue' in fin.columns and len(fin) > 0 else None
        rev_prev = fin['Total Revenue'].iloc[1] if 'Total Revenue' in fin.columns and len(fin) > 1 else None
        
        # Función auxiliar de crecimiento porcentual
        def crecimiento_pct(a, b):
            return ((a - b) / abs(b)) * 100 if a is not None and b is not None and b != 0 else None

        rev_growth = crecimiento_pct(rev_now, rev_prev)

        fcf_now, fcf_prev = None, None
        if 'Free Cash Flow' in cf.columns and len(cf) > 1:
            fcf_now, fcf_prev = cf['Free Cash Flow'].iloc[0], cf['Free Cash Flow'].iloc[1]
        elif 'Operating Cash Flow' in cf.columns and 'Capital Expenditure' in cf.columns and len(cf) > 1:
            fcf_now = cf['Operating Cash Flow'].iloc[0] + cf['Capital Expenditure'].iloc[0]
            fcf_prev = cf['Operating Cash Flow'].iloc[1] + cf['Capital Expenditure'].iloc[1]

        fcf_growth = crecimiento_pct(fcf_now, fcf_prev)

        ebitda = info.get('ebitda', 0)
        total_debt = info.get('totalDebt', 0)
        total_cash = info.get('totalCash', 0)
        
        nd_ebitda = (total_debt - total_cash) / ebitda if ebitda and ebitda > 0 else None
        roe = (info.get('returnOnEquity') or 0) * 100
        peg = info.get('pegRatio', 0) or 0

        score = 0
        st_rev = "❌"
        if rev_growth is not None and rev_growth > 0: score += 20; st_rev = "✅"

        st_fcf = "❌"
        if fcf_growth is not None and fcf_growth > 0: score += 20; st_fcf = "✅"

        st_roe = "❌"
        if roe > 15: score += 20; st_roe = "✅"

        st_debt = "❌"
        if nd_ebitda is not None and nd_ebitda < 2.5: score += 20; st_debt = "✅"
        elif nd_ebitda is None: st_debt = "⚪ (N/A)"

        st_peg = "🟡"
        if 0 < peg < 2.0: score += 10; st_peg = "✅"

        st_up = "🟡"
        if upside > 15: score += 10; st_up = "✅"

        if score >= 85: veredicto = "💎 MUY ATRACTIVO: Activo premium con todo a favor."
        elif 65 <= score < 85: veredicto = "✅ ATRACTIVO: Sólido, buscar confirmación técnica."
        elif 45 <= score < 65: veredicto = "🟡 NEUTRAL: Ver evolución de márgenes o deuda."
        else: veredicto = "🚨 EVITAR: Fundamentos débiles o riesgo alto."

        return {
            "rev_growth": rev_growth, "st_rev": st_rev,
            "fcf_growth": fcf_growth, "st_fcf": st_fcf,
            "roe": roe, "st_roe": st_roe,
            "nd_ebitda": nd_ebitda, "st_debt": st_debt,
            "peg": peg, "st_peg": st_peg,
            "upside": upside, "st_up": st_up,
            "score": score, "veredicto": veredicto
        }
    except Exception:
        return None
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
                   "50% técnico (SMA50/200, RSI).")
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
# ------------------------------- TAB 3 ----------------------------------------
with tab3:
    st.subheader("🔍 Generador de Dashboard P/E & Técnico (Estilo Telegram)")
    
    col_t1, col_t2, col_t3, col_t4 = st.columns([2, 1, 1, 1])
    with col_t1:
        ticker_input = normalizar_ticker(st.text_input("Ingresá un Ticker para analizar:", value="AAPL"))
    with col_t2:
        pe_bajo = st.number_input("P/E Bajo", value=21.0, step=1.0)
    with col_t3:
        pe_medio = st.number_input("P/E Medio", value=30.0, step=1.0)
    with col_t4:
        pe_alto = st.number_input("P/E Alto", value=40.0, step=1.0)

    if st.button("📊 Generar Dashboard Completo") and ticker_input:
        with st.spinner(f"Calculando bandas de P/E, Fuerza Relativa y EPS para {ticker_input}..."):
            foto, resultado_tecnico = generar_dashboard_grafico(ticker_input, pe_bajo, pe_medio, pe_alto)
            checklist = check_pm_pro(ticker_input)
            
        if foto is not None:
            st.session_state["dashboard_pe"] = {
                "ticker": ticker_input,
                "foto": foto,
                "stats": resultado_tecnico,
                "checklist": checklist
            }
        else:
            st.session_state.pop("dashboard_pe", None)
            st.error(f"No se pudo generar el gráfico: {resultado_tecnico}")

    # Mantener estado en session_state para que no desaparezca al interactuar
    if "dashboard_pe" in st.session_state:
        d = st.session_state["dashboard_pe"]
        st.markdown(f"### 📊 Reporte Técnico: `{d['ticker']}`")
        
        # Métricas de resumen rápido
        st_data = d["stats"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P/E Actual", f"{st_data['pe']:.2f}x")
        m2.metric("RSI (14)", f"{st_data['rsi']:.2f}")
        m3.metric("Disp. EMA 200", f"{st_data['disp']:+.2f}%", f"Prom. Neg: {st_data['d_neg']:.2f}%")
        m4.metric("Fuerza Relativa (SPY)", "🟢 LIDERANDO" if st_data["rs_lidera"] else "🔴 REZAGADO")
        
        # Mostrar el gráfico generado con Matplotlib
        st.image(d["foto"], use_container_width=True)
        
        # Mostrar Checklist Fundamental si está disponible
        if d["checklist"]:
            chk = d["checklist"]
            st.markdown("---")
            st.subheader("📋 Checklist Fundamental de Inversión")
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                st.write(f"• **Crec. Ingresos YoY:** `{chk['rev_growth']:+.2f}%`" if chk['rev_growth'] is not None else "• **Crec. Ingresos YoY:** `N/D`")
                st.write(f"• **Crec. FCF YoY:** `{chk['fcf_growth']:+.2f}%`" if chk['fcf_growth'] is not None else "• **Crec. FCF YoY:** `N/D`")
            with c_f2:
                st.write(f"• **ROE Actual:** `{chk['roe']:.2f}%` {chk['st_roe']}")
                st.write(f"• **Net Debt / EBITDA:** `{chk['nd_ebitda']:.2f}x` {chk['st_debt']}" if chk['nd_ebitda'] is not None else "• **Net Debt / EBITDA:** `N/D`")
            with c_f3:
                st.write(f"• **PEG Ratio:** `{chk['peg']:.2f}` {chk['st_peg']}")
                st.write(f"• **Upside Target:** `{chk['upside']:.2f}%` {chk['st_up']}")
            
            st.markdown(f"**🏆 SCORE FUNDAMENTAL:** `{chk['score']}/100`")
            st.markdown(f"**💬 VEREDICTO:** {chk['veredicto']}")
