import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from typing import List


#downlad price from csv 
df = pd.read_csv('dnp_d.csv')
close = df['Zamkniecie']



def EMA(data: Sequence[float], span: int) -> np.ndarray:
    ema = np.zeros(len(data), dtype=float) 
    alpha: float = 2/(span + 1)
    alpha = 2 / (span + 1)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    
    return ema


def find_sell_points(macd: np.ndarray,signal: np.ndarray) -> List[int]:
    sellPoint: List[int] = []
    for i in range(1,len(macd)):
        if macd[i] < signal[i] and macd[i-1] > signal[i-1]:
            sellPoint.append(i)
    return sellPoint

def find_buy_points(macd: np.ndarray,signal: np.ndarray) -> List[int]:
    buyPoint: List[int] = []
    for i in range(1,len(macd)):
        if macd[i] > signal[i] and macd[i-1] < signal[i-1]:
            buyPoint.append(i)
    return buyPoint


def MACD(data: Sequence[float]) -> np.ndarray:
    ema12 = EMA(close,12)
    ema26 = EMA(close,26)
    macd = np.zeros(len(close), dtype=float) 
    for i in range(0,len(close)):
        macd[i] = ema12[i]-ema26[i]
    return macd


def simulate_capital(sellPoint: np.ndarray,buyPoint: np.ndarray,close: Sequence[float],starting_capital:float) -> List[float]:
    arrayShift: int = 0
    capital = [starting_capital]
    stock:float = 0
    temp:int = 0
    if sellPoint[0] < buyPoint[0]:
        arrayShift = 1
        temp = 1

    for i in range(1,len(close)-1):
        stock = capital[i-1]
        if i == sellPoint[temp]:
            stock = stock + close[sellPoint[temp]] - close[buyPoint[temp-arrayShift]]
            if temp + 1 < len(sellPoint):
                temp = temp + 1
        capital.append(stock)
    
    return capital



macd = MACD(close)
signal = EMA(macd,9)
sellPoint = find_sell_points(macd,signal)
buyPoint = find_buy_points(macd,signal)
capital = simulate_capital(sellPoint,buyPoint,close,1000)

range_macd = int(len(macd)/2)
range_signal = int(len(signal)/2)
range_buypoints = int(len(buyPoint)/2)
range_sellpoints = int(len(sellPoint)/2)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

start0 = 105
end0   = 200 

axs[0, 0].plot(range(start0, end0 + 1), macd[start0 : end0 + 1], label='MACD')
axs[0, 0].plot(range(start0, end0 + 1), signal[start0 : end0 + 1], label='Signal')
buyPoint0 = [i for i in buyPoint if start0 <= i <= end0]
sellPoint0 = [i for i in sellPoint if start0 <= i <= end0]
axs[0, 0].scatter(buyPoint0, [macd[i] for i in buyPoint0], color='green', marker='^', s=20, label='Kupno')
axs[0, 0].scatter(sellPoint0, [macd[i] for i in sellPoint0], color='red', marker='v', s=20, label='Sprzedaż')
axs[0, 0].legend()
axs[0, 0].set_title(f'MACD/Signal {start0}-{end0}')

axs[1, 0].plot(range(start0, end0 + 1), close[start0 : end0 + 1], label='MACD')
buyPoint0 = [i for i in buyPoint if start0 <= i <= end0]
sellPoint0 = [i for i in sellPoint if start0 <= i <= end0]
axs[1, 0].scatter(buyPoint0, [close[i] for i in buyPoint0], color='green', marker='^', s=20, label='Kupno')
axs[1, 0].scatter(sellPoint0, [close[i] for i in sellPoint0], color='red', marker='v', s=20, label='Sprzedaż')
axs[1, 0].legend()
axs[1, 0].set_title(f'Price {start0}-{end0}')


start1 = 1000
end1   = 1150

axs[0, 1].plot(range(start1, end1 + 1), macd[start1 : end1 + 1], label='MACD')
axs[0, 1].plot(range(start1, end1 + 1), signal[start1 : end1 + 1], label='Signal')
buyPoint1 = [i for i in buyPoint if start1 <= i <= end1]
sellPoint1 = [i for i in sellPoint if start1 <= i <= end1]
axs[0, 1].scatter(buyPoint1, [macd[i] for i in buyPoint1], color='green', marker='^', s=20, label='Kupno')
axs[0, 1].scatter(sellPoint1, [macd[i] for i in sellPoint1], color='red', marker='v', s=20, label='Sprzedaż')
axs[0, 1].legend()
axs[0, 1].set_title(f'MACD/Signal {start1}-{end1}')

axs[1, 1].plot(range(start1, end1 + 1), close[start1 : end1 + 1], label='MACD')
buyPoint0 = [i for i in buyPoint if start1 <= i <= end1]
sellPoint0 = [i for i in sellPoint if start1 <= i <= end1]
axs[1, 1].scatter(buyPoint1, [close[i] for i in buyPoint1], color='green', marker='^', s=20, label='Kupno')
axs[1, 1].scatter(sellPoint1, [close[i] for i in sellPoint1], color='red', marker='v', s=20, label='Sprzedaż')
axs[1, 1].legend()
axs[1, 1].set_title(f'Price {start1}-{end1}')


plt.tight_layout() 
plt.show()


fig, axs = plt.subplots(2,1,figsize=(12, 8))

axs[0].plot(range(len(close)), close, label='Price')
axs[0].scatter(
    buyPoint,
    [close[i] for i in buyPoint],
    color='green', marker='^', s=20,
    label='Kupno'
)
axs[0].scatter(
    sellPoint,
    [close[i] for i in sellPoint],
    color='red', marker='v', s=20,
    label='Sprzedaż'
)
axs[0].legend()

axs[1].plot(range(len(capital)), capital , label='capital')
axs[1].legend()

plt.legend()
plt.tight_layout() 
plt.show()
