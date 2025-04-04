import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, List

# Load data from CSV file and extract the 'Close' prices
df = pd.read_csv('dnp_d.csv')
close = df['Zamkniecie']  # 'Zamkniecie' means 'Close' in Polish


def EMA(data: Sequence[float], span: int) -> np.ndarray:
    """Calculate the Exponential Moving Average (EMA) for a given data series."""
    ema = np.zeros(len(data), dtype=float)
    alpha = 2 / (span + 1)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema


def MACD(data: Sequence[float]) -> np.ndarray:
    """Calculate the MACD indicator as the difference between 12 and 26-period EMAs."""
    ema12 = EMA(close, 12)
    ema26 = EMA(close, 26)
    macd = ema12 - ema26
    return macd


def find_buy_points(macd: np.ndarray, signal: np.ndarray) -> List[int]:
    """Detect buy points (MACD crosses above signal)."""
    buy_points = []
    for i in range(1, len(macd)):
        if macd[i] > signal[i] and macd[i - 1] < signal[i - 1]:
            buy_points.append(i)
    return buy_points


def find_sell_points(macd: np.ndarray, signal: np.ndarray) -> List[int]:
    """Detect sell points (MACD crosses below signal)."""
    sell_points = []
    for i in range(1, len(macd)):
        if macd[i] < signal[i] and macd[i - 1] > signal[i - 1]:
            sell_points.append(i)
    return sell_points


def simulate_capital(sell_points: List[int], buy_points: List[int],
                     close: Sequence[float], starting_capital: float) -> List[float]:
    """
    Simulate capital evolution by performing buy/sell operations based on detected points.
    """
    shift = 0
    capital = [starting_capital]
    temp = 0

    if sell_points[0] < buy_points[0]:
        shift = 1
        temp = 1

    for i in range(1, len(close) - 1):
        balance = capital[i - 1]
        if i == sell_points[temp]:
            profit = close[sell_points[temp]] - close[buy_points[temp - shift]]
            balance += profit
            if temp + 1 < len(sell_points):
                temp += 1
        capital.append(balance)

    return capital


# Compute MACD and SIGNAL lines
macd = MACD(close)
signal = EMA(macd, 9)

# Identify buy/sell points
sell_points = find_sell_points(macd, signal)
buy_points = find_buy_points(macd, signal)

# Simulate capital starting from 1000 units
capital = simulate_capital(sell_points, buy_points, close, 1000)

# Plotting results for selected intervals
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# First window
start0, end0 = 105, 200
axs[0, 0].plot(range(start0, end0 + 1), macd[start0:end0 + 1], label='MACD')
axs[0, 0].plot(range(start0, end0 + 1), signal[start0:end0 + 1], label='Signal')
buy_points0 = [i for i in buy_points if start0 <= i <= end0]
sell_points0 = [i for i in sell_points if start0 <= i <= end0]
axs[0, 0].scatter(buy_points0, [macd[i] for i in buy_points0], color='green', marker='^', label='Buy')
axs[0, 0].scatter(sell_points0, [macd[i] for i in sell_points0], color='red', marker='v', label='Sell')
axs[0, 0].legend()
axs[0, 0].set_title(f'MACD/Signal {start0}-{end0}')

axs[1, 0].plot(range(start0, end0 + 1), close[start0:end0 + 1], label='Price')
axs[1, 0].scatter(buy_points0, [close[i] for i in buy_points0], color='green', marker='^', label='Buy')
axs[1, 0].scatter(sell_points0, [close[i] for i in sell_points0], color='red', marker='v', label='Sell')
axs[1, 0].legend()
axs[1, 0].set_title(f'Price {start0}-{end0}')

# Second window
start1, end1 = 1000, 1150
axs[0, 1].plot(range(start1, end1 + 1), macd[start1:end1 + 1], label='MACD')
axs[0, 1].plot(range(start1, end1 + 1), signal[start1:end1 + 1], label='Signal')
buy_points1 = [i for i in buy_points if start1 <= i <= end1]
sell_points1 = [i for i in sell_points if start1 <= i <= end1]
axs[0, 1].scatter(buy_points1, [macd[i] for i in buy_points1], color='green', marker='^', label='Buy')
axs[0, 1].scatter(sell_points1, [macd[i] for i in sell_points1], color='red', marker='v', label='Sell')
axs[0, 1].legend()
axs[0, 1].set_title(f'MACD/Signal {start1}-{end1}')

axs[1, 1].plot(range(start1, end1 + 1), close[start1:end1 + 1], label='Price')
axs[1, 1].scatter(buy_points1, [close[i] for i in buy_points1], color='green', marker='^', label='Buy')
axs[1, 1].scatter(sell_points1, [close[i] for i in sell_points1], color='red', marker='v', label='Sell')
axs[1, 1].legend()
axs[1, 1].set_title(f'Price {start1}-{end1}')

plt.tight_layout()
plt.show()

# Final plot: full time series and capital growth
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].plot(close, label='Price')
axs[0].scatter(buy_points, [close[i] for i in buy_points], color='green', marker='^', label='Buy')
axs[0].scatter(sell_points, [close[i] for i in sell_points], color='red', marker='v', label='Sell')
axs[0].legend()

axs[1].plot(capital, label='Capital')
axs[1].legend()

plt.tight_layout()
plt.show()
