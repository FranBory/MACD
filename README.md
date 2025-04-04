# 📈 MACD-Based Stock Analysis & Simulation

This project implements a trading signal detection system using the **MACD (Moving Average Convergence Divergence)** indicator. It is developed as part of a numerical methods assignment and allows users to detect buy/sell signals, simulate capital growth, and visualize trading strategy performance based on historical stock data.

## 📚 About MACD

MACD is a technical indicator used to identify trend reversals and momentum in financial markets. It is based on the difference between two **Exponential Moving Averages (EMAs)**:
- `MACD = EMA(12) - EMA(26)`
- A **Signal line** is calculated as an EMA(9) of the MACD

### 🔁 Buy/Sell Strategy
- **Buy**: when MACD crosses above the Signal line
- **Sell**: when MACD crosses below the Signal line

These crossovers are interpreted as potential entry and exit points.

---

## 💻 Features

- ✅ Manual implementation of EMA, MACD, and Signal line (no external finance libs!)
- ✅ Detection of buy/sell points
- ✅ Simulation of capital growth from an initial investment
- ✅ Multiple visualization plots with trading markers
- ✅ Configurable and reusable for other CSV datasets
