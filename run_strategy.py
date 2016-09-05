import numpy as np
import pandas as pd
import datetime
import json

from sqlalchemy import create_engine
from pandas.io import sql
from strategy import MovingAverageCrossStrategy, MarketOnClosePortfolio, MACDStrategy

engine = create_engine('mysql://root:testdb@127.0.0.1/tstest?charset=utf8')

#create start date, end date and symbol list
cnx = engine.raw_connection() # option-2
eqs = sql.read_sql("SELECT * FROM equ_info_a", cnx)
cnx.close()
sd = datetime.date.today() - datetime.timedelta(days=250)
ed = datetime.date.today()
symbols = eqs.secID

#function to get data from database
def get_bars(ids,sd,ed):
    ids = tuple(ids.apply(str))
    cnx = engine.raw_connection() # option-2
    eodp = pd.DataFrame()
    #create query pull price history for all stocks
    query = "SELECT secID,tradeDate, openPrice, highestPrice, lowestPrice, closePrice, turnoverVol  FROM mkt_price where secID in %r and tradeDate between %r and %r" %(ids, str(sd),str(ed))
    edop = sql.read_sql(query, cnx)    
        
    return edop

#reshape data
data = get_bars(symbols.head(10),sd,ed)
bars = data.pivot('tradeDate','secID')['closePrice']

#get signals and indicators
sig1 = MovingAverageCrossStrategy( bars, short_window=10, long_window=30)
sig2 = MACDStrategy(bars)

#get trading positions
trade1 = sig1.generate_trades()
trade2 = sig2.generate_trades()

#backtest strategy

portfolio = MarketOnClosePortfolio(bars,trade1,initial_capital=100000.0)
portfolio2 = MarketOnClosePortfolio(bars,trade2,initial_capital=100000.0)

pf = portfolio.backtest_portfolio()
pf2 = portfolio2.backtest_portfolio()


#plot returns
import matplotlib.pyplot as plt
    
fig = plt.figure(figsize=(15, 20))
#fig = plt.figure()
fig.patch.set_facecolor('white')     # Set the outer colour to white
ax1 = fig.add_subplot(411,  ylabel='Price in $')

pf['holdings']['total'].plot(ax=ax1, lw=2.)
pf2['holdings']['total'].plot(ax=ax1, lw=2.)

ax2 = fig.add_subplot(412, ylabel='Portfolio value in $')
pf['holdings']['returns'].cumsum().plot(ax=ax2)
pf2['holdings']['returns'].cumsum().plot(ax=ax2)