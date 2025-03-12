import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('dnp_d.csv')
#df = pd.read_csv('ubisoft.csv')
close = df['Zamkniecie']



def EMA(data,span):
    ema = np.zeros(len(data), dtype=float) 
    #ema = np.zeros(span, dtype=float)
    a = 2/(span + 1)
    x = 1 - a

    for i in range(0,span):
        emaL = data[i]
        emaM = 1
        for j in range(1,i+1):
            emaL = emaL + data[i-j] * x**j 
            emaM = emaM + x**j
        ema[i] = emaL / emaM

    for i in range(span,len(data)):
        emaL = data[i]
        emaM = 1
        for j in range(1,span+1):
            emaL = emaL + data[i-j] * x**j 
            emaM = emaM + x**j
        ema[i] = emaL / emaM
    
    return ema
    

#print(close.head())
ema12 = EMA(close,12)
ema26 = EMA(close,26)
macd = np.zeros(len(close), dtype=float) 
for i in range(0,len(close)):
    macd[i] = ema12[i]-ema26[i]

# print(macd)

signal = EMA(macd,9)


sellPoint = []
buyPoint = []

for i in range(1,len(macd)):
    if macd[i] < signal[i] and macd[i-1] > signal[i-1]:
            sellPoint.append(i)
    if macd[i] > signal[i] and macd[i-1] < signal[i-1]:
            buyPoint.append(i)



capital = [1000]
temp = 1


for i in range(1,len(close)-1 ):
    stock = capital[i-1]
    # print(i)
    if i == sellPoint[temp]:
         print(i,sellPoint[temp] - buyPoint[temp-1])
         stock = stock + close[sellPoint[temp]] - close[buyPoint[temp-1]]
         if temp + 1 < len(sellPoint):
            temp = temp + 1
    
    capital.append(stock)



range_macd = int(len(macd)/2)
range_signal = int(len(signal)/2)
range_buypoints = int(len(buyPoint)/2)
range_sellpoints = int(len(sellPoint)/2)
# print(range_buypoints)
# print(range_sellpoints)

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
