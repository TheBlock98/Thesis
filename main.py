import pandas as pd
from finta import TA
import pandas_ta as pta
import seaborn as sns
import matplotlib.pyplot as plt

dataSet = pd.read_csv("BTCUSDT-spot-1h.csv", index_col="date", parse_dates=True)

SpxData = pd.read_csv("SPX.csv",parse_dates=True)
SpxData.rename(columns={'Date': 'date'}, inplace=True)
SpxData.set_index('date', inplace=True)

DxyData = pd.read_csv("DXY.csv",parse_dates=True)
DxyData.rename(columns={'Date': 'date'}, inplace=True)
DxyData.set_index('date', inplace=True)

DIXGEXData = pd.read_csv("DIXGEX.csv",parse_dates=True)
DIXGEXData.rename(columns={'Date': 'date'}, inplace=True)
DIXGEXData.set_index('date', inplace=True)
dataSet = dataSet.iloc[1:]

# Feature extraction using both `finta` and `pandas_ta`

# Exponential Moving Average (EMA) features
dataSet["EMA_20"] = TA.EMA(dataSet, 20)
dataSet["EMA_50"] = TA.EMA(dataSet, 50)
dataSet["EMA_200"] = TA.EMA(dataSet, 200)
dataSet["hlc3"] = (dataSet["high"] + dataSet["low"] + dataSet["close"]) / 3



# Average True Range (ATR) features
dataSet["ATR"] = TA.ATR(dataSet, 24)  # Daily ATR
dataSet["ATR_168"] = TA.ATR(dataSet, 168)  # Weekly ATR

# Relative Strength Index (RSI)
dataSet["RSI"] = TA.RSI(dataSet, 14)





# Moving Average Convergence Divergence (MACD)
macd_data = TA.MACD(dataSet)
dataSet["MACD_line"] = macd_data["MACD"]
dataSet["MACD_signal"] = macd_data["SIGNAL"]

# Hull Moving Average (HMA) features
dataSet["HMA_9"] = pta.hma(dataSet["close"], 9)
dataSet["HMA_16"] = pta.hma(dataSet["close"], 16)

# Bollinger Bands (BB) features
bbands_10_data = pta.bbands(dataSet["close"], length=10, std=1.5)
bbands_20_data = pta.bbands(dataSet["close"], length=20, std=2)
bbands_50_data = pta.bbands(dataSet["close"], length=50, std=2.5)

# Join BB data to the main dataset
dataSet = dataSet.join(bbands_10_data)
dataSet = dataSet.join(bbands_20_data)
dataSet = dataSet.join(bbands_50_data)

# More indicators
dataSet["ebws"] = pta.ebsw(dataSet["close"])
fisher_data = pta.fisher(dataSet["high"], dataSet["low"],16)
dataSet["FISHER"] = fisher_data.iloc[:, 0]
dataSet["FISHERT"] = fisher_data.iloc[:, 1]

# Detrended Price Oscillator (DPO)
dataSet["DPO14"] = pta.dpo(close=dataSet['close'], length=14, centered=False)




# pivot 


dataSetD = dataSet.resample('D').agg({
    'open': 'first',   # Primo valore del giorno
    'high': 'max',     # Massimo valore delle 24 ore
    'low': 'min',      # Minimo valore delle 24 ore
    'close': 'last'    # Ultimo valore del giorno
})

previous_day = dataSetD.shift(1)
pp = (dataSetD['high'] + dataSetD['low'] + dataSetD['close']) / 3

r4 = pp + ((previous_day["high"] - previous_day["low"]) * 1.382)
r3 = pp + ((previous_day["high"] - previous_day["low"]) * 1)
r2 = pp + ((previous_day["high"] - previous_day["low"]) * 0.618)
r1 = pp + ((previous_day["high"] - previous_day["low"]) * 0.382)

s1 = pp - ((previous_day["high"] - previous_day["low"]) * 0.382)
s2 = pp - ((previous_day["high"] - previous_day["low"]) * 0.618)
s3 = pp - ((previous_day["high"] - previous_day["low"]) * 1)
s4 = pp - ((previous_day["high"] - previous_day["low"]) * 1.382)

# Combine the pivot levels into a single DataFrame
pivot_data = pd.DataFrame({
            'pivot': pp,
            's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            'r4': r4
        },index=dataSetD.index)
# Merge the pivot data for each day into each row of hourly data that falls under that day
dataSet = pd.merge_asof(dataSet.sort_index(), pivot_data.sort_index(), left_index=True, right_index=True, direction='backward')


# Future Line of Demarcation (FLD)
fldData = pd.DataFrame(index=dataSetD.index)
for period in [35, 18, 10]:  # Organize the periods to calculate FLD
    src = (dataSetD['high'] + dataSetD['low']) / 2
    fldData[f'FLD{period}'] = src.shift(period)
dataSet = pd.merge_asof(dataSet.sort_index(), fldData.sort_index(), left_index=True, right_index=True, direction='backward')




# Volume Weighted Average Price (VWAP)
dataSet["D-VWAP"] = pta.vwap(high=dataSet["high"], low=dataSet["low"], close=dataSet["close"], volume=dataSet["volume"])
dataSet["W-VWAP"] = pta.vwap(high=dataSet["high"], low=dataSet["low"], close=dataSet["close"], volume=dataSet["volume"],anchor = "W")

# Volume Weighted Moving Average (VWMA)
dataSet["VWMA"] = pta.vwma(close=dataSet["close"], volume=dataSet["volume"], length=20)

# Easy of market move
dataSet["EOM"] = pta.eom(high=dataSet["high"], low=dataSet["low"], close=dataSet["close"], volume=dataSet["volume"])

# Money flow index:
dataSet["MFI"] = pta.mfi(high=dataSet["high"], low=dataSet["low"], close=dataSet["close"], volume=dataSet["volume"], length=14)

# Fourier discrete transofmration (FT) --> MFI



# inter-market relation (IMR)
SpxData['SPX500-DailyReturn%'] = SpxData['Close'].pct_change() * 100
SpxData['SPXEMA20']= TA.EMA(SpxData, 20)

DxyData['DXY-DailyReturn%'] = DxyData['Close'].pct_change() * 100
DxyData['DXYEMA200']= TA.EMA(DxyData, 200)


dataSetIMR = dataSet # in this way we can test if IMR is useful or not


SpxData.index = pd.to_datetime(SpxData.index)
DxyData.index = pd.to_datetime(DxyData.index)
DIXGEXData.index = pd.to_datetime(DIXGEXData.index)
SpxData.index = SpxData.index.tz_localize(None)
DxyData.index = DxyData.index.tz_localize(None)
DIXGEXData.index = DIXGEXData.index.tz_localize(None)
dataSetIMR.index = dataSetIMR.index.tz_localize(None) if dataSetIMR.index.tz is not None else dataSetIMR.index


dataSetIMR = pd.merge_asof(dataSetIMR.sort_index(), SpxData[['SPX500-DailyReturn%','SPXEMA20']], left_index=True, right_index=True, direction='backward')
dataSetIMR = pd.merge_asof(dataSetIMR.sort_index(), DxyData[['DXY-DailyReturn%','DXYEMA200']], left_index=True, right_index=True, direction='backward')
dataSetIMR = pd.merge_asof(dataSetIMR.sort_index(), DIXGEXData[['dix','gex']], left_index=True, right_index=True, direction='backward')



# Find the index of the first non-NaN row across all columns
first_valid_index = dataSet.dropna().index[0]

dataSetCleaned = dataSet.loc[first_valid_index:]
dataSetIMRCleaned = dataSetIMR.loc[first_valid_index:]

print(dataSetIMR[['SPXEMA20', 'DXYEMA200']].dtypes)
print(dataSetIMRCleaned[['SPXEMA20', 'DXYEMA200']])

dataSetIMR.fillna(0, inplace=True)  # ho le chiusure del weekand così risolvo il problema


dataSetIMRCleaned = dataSetIMRCleaned.select_dtypes(include=['float64', 'int64'])
dataSetIMR['SPX500-DailyReturn%'] = pd.to_numeric(dataSetIMR['SPX500-DailyReturn%'], errors='coerce')
dataSetIMR['DXY-DailyReturn%'] = pd.to_numeric(dataSetIMR['DXY-DailyReturn%'], errors='coerce')

print(dataSetIMR.head())






# salvare il dataSetIMR
dataSetIMRCleaned.to_csv('dataSetIMRCleaned.csv')



