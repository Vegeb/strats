import numpy as np
import pandas as pd
import datetime
import json

from sqlalchemy import create_engine
from pandas.io import sql
from strategy import *



#function to get 1 year MACD of a single stock
def get_macd(ids,short_window=12, long_window=26, signal_window=9):
    #initialize
    engine = create_engine('mysql://root:testdb@127.0.0.1/tstest?charset=utf8')
    sd = datetime.date.today() - datetime.timedelta(days=365)
    ed = datetime.date.today()
    
    #db connection
    cnx = engine.raw_connection() # option-2
    eodp = pd.DataFrame()
    #create query pull price history for all stocks
    query = "SELECT secID,tradeDate, openPrice, highestPrice, lowestPrice, closePrice, turnoverVol  FROM mkt_price where secID = %r and tradeDate between %r and %r" %(ids, str(sd),str(ed))
    edop = sql.read_sql(query, cnx)  
    cnx.close()
    #reshape
    bars = edop.pivot('tradeDate','secID')['closePrice']
    sig = MACDStrategy(bars,short_window, long_window, signal_window) 
    
    result = pd.concat([sig.signals['MACD'],sig.signals['MACD'],sig.signals['MACD']],axis=1,keys=['macd','signal','divergence'])
    result = result.dropna(how='all')
    
    return result.to_json(date_format='iso')



