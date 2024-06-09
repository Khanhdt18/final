# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:25:01 2020

@author: dangt
"""

import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import scipy.optimize as sco
import xlrd

start = datetime.datetime(2014, 5, 1)
end = datetime.datetime(2015, 5, 31)
df1 = pd.read_excel(r'C:\Users\dangt\Desktop\house_data.xls','Sheet2')
symbols=np.array(df1.iloc[:,0])
noa = len(symbols)

data = pd.DataFrame()
dfk = pd.DataFrame()
for sym in symbols:
    dfk = web.get_data_yahoo(sym, start, end, interval='m')
    dfk.reset_index(inplace=True)
    dfk.set_index("Date", inplace=True)

    data[sym] = dfk['Close']

realestate_asset_value = pd.Series([71.830002,67.239998,58.000000,61.950001,69.550003,61.650002,49.450001,60.900002,64.650002,68.199997,
                                   58.250000,58.000000,49.450001])
data["REAL ESTATE"] = np.nan
for i in range(len(data)):
    data.iloc[i,noa] = realestate_asset_value[i]
noa = noa+1
symbols=np.append(symbols, "REAL ESTATE" )

rets = np.log(data / data.shift(1))
(data / data.iloc[0]*100).plot(figsize=(8, 5))


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(rets.cov() * 12, vmin=0, vmax=0.4)
fig.colorbar(cax)
ticks = np.arange(0,len(symbols),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(symbols)
ax.set_yticklabels(symbols)
plt.show()

weights = np.random.random(noa)
weights /= np.sum(weights)

prets = []
pvols = []
for p in range (500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean() * weights) * 12)
    pvols.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 12, weights))))
prets = np.array(prets)
pvols = np.array(pvols)


def statistics(weights):
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 12
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 12, weights)))
    return np.array([pret, pvol, pret / pvol])

def min_func_sharpe(weights):
    return -statistics(weights)[2]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

bnds = tuple((0, 1) for x in range(noa))

opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], 
                    method='SLSQP', bounds=bnds, constraints=cons)

def min_func_variance(weights):
    return statistics(weights)[1] ** 2

optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP', bounds=bnds,constraints=cons)

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
# random portfolio composition
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0], 'r*', markersize=15.0)
# portfolio with highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'y*', markersize=15.0)
# minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')



