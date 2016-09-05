# -*- coding: utf-8 -*-
"""
Created on Mon May 09 08:01:03 2016

@author: Administrator
"""

import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pandas.io import sql



class MovingAverageCrossStrategy(object):


    def __init__(self, bars, short_window=100, long_window=400):
        
        self.bars = bars

        self.short_window = short_window
        self.long_window = long_window
        self.signals = self.generate_signals()

    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = {}

        # Create the set of short and long simple moving averages over the 
        # respective periods
        signals['short_mavg'] = self.bars.rolling(window=self.short_window).mean()
        signals['long_mavg'] = self.bars.rolling(window=self.long_window).mean()


        return signals
    
    def generate_trades(self):
        trades = {}
        trades['signal'] = pd.DataFrame(np.nan,index=self.bars.index, columns=self.bars.columns)
        # Create a 'signal' (invested or not invested) when the short moving average crosses the long
        # moving average, but only for the period greater than the shortest moving average window
        trades['signal'][self.long_window:] = 0
        trades['signal'][self.long_window:] = trades['signal'][self.long_window:].where(self.signals['short_mavg'][self.long_window:]<= self.signals['long_mavg'][self.long_window:],1)

        # Take the difference of the signals in order to generate actual trading orders
        trades['positions'] = trades['signal'].diff()   
        return trades
        

class MACDStrategy(object):


    def __init__(self, bars, short_window=12, long_window=26, signal_window=9):
        
        self.bars = bars

        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.signals = self.generate_signals()

    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = {}
            

        # Create the set of short and long exponential moving averages over the 
        # respective periods
        signals['short'] = self.bars.ewm(span = self.short_window , min_periods=self.long_window-1).mean()
        signals['long'] = self.bars.ewm(span = self.long_window , min_periods=self.long_window-1).mean()
        signals['MACD'] = signals['short'] - signals['long']
        signals['MACDsign'] = signals['MACD'].ewm(span = self.signal_window , min_periods=self.long_window-1).mean()
        signals['MACDdiff'] = signals['MACD'] - signals['MACDsign']

        
        return signals  
    
    def generate_trades(self):
        trades = {}
        trades['signal'] = pd.DataFrame(np.nan,index=self.bars.index, columns=self.bars.columns)
        
        # Create a 'signal' (invested or not invested) when the short moving average crosses the long
        # moving average, but only for the period greater than the shortest moving average window
        trades['signal'][self.long_window:] = 0
        trades['signal'] = trades['signal'].where(self.signals['MACD'] <= self.signals['MACDsign'], 1.0)   

        # Take the difference of the signals in order to generate actual trading orders
        trades['positions'] = trades['signal'].diff()   
        return trades


class BOLLStrategy(object):
    #takes input bars, which contains eod close price
    #bars is a dataframe with date as index, stock ids as column names


    def __init__(self, bars, window=50):
        
        self.bars = bars
        self.window = window
        self.stats = self.generate_stats()
        self.signals = self.generate_signals()

    def generate_stats(self):
        #Returns the DataFrame of symbols containing the statistical indicators
        
        stats = {}
            

        # generate stats
        stats['ma'] = self.bars.rolling(window=self.window).mean()
        stats['std'] = self.bars.rolling(window=self.window).std()
        stats['upper'] = stats['ma'] + 2*stats['std']
        stats['lower'] = stats['ma'] - 2*stats['std']
        stats['bw'] = 4*stats['std'] / stats['ma']
        stats['pct_b'] = (self.bars - stats['lower']) / (4*stats['std'])

        
        return stats  
    
    def generate_signals(self):
        #generate trading signals to go long or short or hold

        #initiate signal frame
        trades = {}
        trades['signal'] = pd.DataFrame(np.nan,index=self.bars.index, columns=self.bars.columns)
        trades['signal'][self.window:] = 0

        # Create a 'signal' 1 when the price crosses upper band
        trades['signal'][(self.bars > self.stats['upper']) & (self.bars > self.stats['ma'])] = 1.0

        # Create a 'signal' -1 when the price crosses lower band
        trades['signal'][(self.bars < self.stats['lower']) & (self.bars < self.stats['ma'])] = -1.0


        # Take the difference of the signals in order to generate actual trading orders
        trades['positions'] = trades['signal'].diff()   
        return trades
    
    
    
class MarketOnClosePortfolio(object):
    """Encapsulates the notion of a portfolio of positions based
    on a set of signals as provided by a Strategy.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, bars, signals, initial_capital=2000000.0):
                
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        post = 100 * self.signals['positions']   # This strategy buys 100 shares
        return post
                    
    def backtest_portfolio(self):
        pf = {}
        pf['holdings'] = self.signals['signal'].mul(self.bars, axis='index')
        pf['cash'] = self.initial_capital - self.positions.mul(self.bars, axis='index').cumsum().sum(axis=1)
        pf['total'] = pf['cash'] + self.positions.cumsum().mul(self.bars, axis='index').sum(axis=1)
        pf['returns'] = pf['total'].pct_change()
        return pf

