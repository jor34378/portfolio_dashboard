import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Test Screener", layout="wide")

st.title("🧪 Modo de Prueba (3 Tickers)")

# Lista estanca de prueba para no fallar con las hojas de cálculo
tickers_prueba = ["AAPL", "MSFT", "GGAL.BA"]

def calcular_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.iloc[-1] if not rsi.empty else 50.0

if st.button("🚀 Probar Descarga de 3 Tickers"):
    resultados = []
    bar = st.progress(0, text="Descargando prueba...")
    
    for i, t in enumerate(tickers_prueba):
        try:
            obj = yf.Ticker(t)
            hist = obj.history(period="6mo")
            if not hist.empty:
                precio = hist["Close"].iloc[-1]
                rsi = calcular_rsi(hist["Close"])
                info = obj.info
                pe = info.get("trailingPE", 0.0) or 0.0
                
                resultados.append({
                    "Ticker": t,
                    "Precio": round(precio, 2),
                    "P/E": round(float(pe), 2),
                    "RSI": round(float(rsi), 2)
                })
        except Exception as e:
            st.write(f"Error con {t}: {e}")
        bar.progress((i + 1) / len(tickers_prueba))
    
    bar.empty()
    
    if resultados:
        df_test = pd.DataFrame(resultados)
        st.success("¡Prueba exitosa! Los datos se descargaron correctamente:")
        st.dataframe(df_test, use_container_width=True)
    else:
      st.error("No se pudo descargar ningún dato de prueba.")
