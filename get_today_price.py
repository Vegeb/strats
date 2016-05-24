# -*- coding: utf-8 -*-
"""
Created on Mon May 23 09:27:16 2016

@author: Administrator
"""

from sqlalchemy import create_engine

import datetime
import tushare as ts
ts.set_token('601be43bd14269103d558400372fc2a18d752d999d4e78a167335353abf79e8e')
engine = create_engine('mysql://root:testdb@127.0.0.1/tstest?charset=utf8')

#equ_info_A = ts.Equity().Equ(equTypeCD='A',field='')
#equ_info_B = ts.Equity().Equ(equTypeCD='B',field='')

#equ_info_A.to_sql('equ_info_a',con=engine, if_exists = 'replace', index = False, index_label = 'secID')
#equ_info_B.to_sql('equ_info_b',con=engine, if_exists = 'replace', index = False, index_label = 'secID')

today = datetime.date.today().strftime('%Y%m%d')
mkt = ts.Market().MktEqud(tradeDate = today,field = '')
try: 
    mkt.to_sql('mkt_price', con=engine, if_exists = 'append', index = False)
except Exception:
    pass