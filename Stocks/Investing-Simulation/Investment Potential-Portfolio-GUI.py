#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
from scipy.stats import linregress
import tkinter as tk
import datetime as dt
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# In[67]:


CPI_CSV = "CPIAUCSL.csv"
CPI_COLUMN = "CPIAUCSL"

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        
        self.symbols = None
        self.portfolio = None
        
    def create_widgets(self):
        # labels
        tk.Label(self, text="Symbols (comma-separated)").grid(row=0)
        tk.Label(self, text="Weights (comma-separated)").grid(row=1)
        tk.Label(self, text="Start Date (yyyy-mm-dd)").grid(row=2)
        tk.Label(self, text="End Date (yyyy-mm-dd)").grid(row=3)
        tk.Label(self, text="Initial Contribution $").grid(row=4)
        tk.Label(self, text="Monthly Contribution $").grid(row=5)
        
        # get inflation
        self.inflation_rate = self.getInflation()

        # input symbols
        self.input_symbols = tk.Entry(self)
        self.input_symbols.grid(row=0, column=1)
        
        # input weights
        self.input_weights = tk.Entry(self)
        self.input_weights.grid(row=1, column=1)
        
        # start date
        self.start_date = tk.Entry(self)
        self.start_date.grid(row=2, column=1)
        
        # end date
        self.end_date = tk.Entry(self)
        self.end_date.grid(row=3, column=1)
        
        # initial contribution
        self.initial = tk.Entry(self)
        self.initial.grid(row=4, column=1)

        # monthly contribution
        self.contrib = tk.Entry(self)
        self.contrib.grid(row=5, column=1)
        
        # submit button
        self.submit = tk.Button(self, text="SUBMIT", fg="red", command=self.run_analysis).grid(row=6, column=0)
        
        # chart buttons
        self.use_cont = tk.IntVar()
        self.chart_cont = tk.Checkbutton(self, text="Monthly Contribution", variable=self.use_cont)
        self.chart_cont.select()
        self.chart_cont.grid(row=6, column=1)
        
        self.use_static = tk.IntVar()
        self.chart_static = tk.Checkbutton(self, text="No Contribution", variable=self.use_static)
        self.chart_static.select()
        self.chart_static.grid(row=6, column=2)

        self.use_inf = tk.IntVar()
        self.chart_inf = tk.Checkbutton(self, text="Inflation", variable=self.use_inf)
        self.chart_inf.select()
        self.chart_inf.grid(row=6, column=3)

        self.input_symbols.insert("10", "SPY, QQQ")
        self.input_weights.insert("10", ".5, .5")
        self.start_date.insert("10", "1990-01-01")
        self.end_date.insert("10", "2020-01-01")
        self.initial.insert("10", "1000")
        self.contrib.insert("10", "100")
        self.error = tk.Message(self, text="")
        self.error.grid(row=10)
        

    def run_analysis(self):
        # gather input data
        symbols = [ el.strip().upper() for el in self.input_symbols.get().split(",") ]
        weights = [ float(el.strip()) for el in self.input_weights.get().split(",") ]
        start_date = self.start_date.get().strip()
        end_date = self.end_date.get().strip()
        init = float(self.initial.get().strip())
        contrib = float(self.contrib.get().strip())
        
        # get porfolio prices (if new input)
        if symbols != self.symbols: # no need to regather data if symbols are the same
            print("Gathering new stock data")
            self.portfolio = self.getClosingPrices(symbols, start_date, end_date)
            self.symbols = symbols
        else:
            print("Using previous data")
            
        # error checking
        if len(weights) != self.portfolio.shape[1]:
            print("Error, unequal inputs")
            self.displayError("Number of symbols and weights do not match")
            return
        if np.sum(weights) != 1:
            print("Error, weights do not sum to 1")
            self.displayError("Weights do not sum to 1")
            return
        self.clearError()
            
        # get portfolio segment from input dates
        start = [ int(el) for el in start_date.split("-") ]
        end = [ int(el) for el in end_date.split("-") ]

        # select portfolio within bounds
        start_date = self.portfolio.index.searchsorted(dt.datetime(start[0], start[1], start[2]))
        end_date = self.portfolio.index.searchsorted(dt.datetime(end[0], end[1], end[2]))
        portfolio = self.portfolio.iloc[start_date:end_date]
        
        # select inflation within bounds
        start_date = self.inflation_rate.index.searchsorted(portfolio.index[0])
        end_date = self.inflation_rate.index.searchsorted(portfolio.index[-1])
        inflation_rate = self.inflation_rate.iloc[start_date:end_date]
        
        # get inflation
        inflation = self.calcReturns(init, 0, inflation_rate.INFLATION_RATE)
                
        # get returns
        returns = self.getReturns(portfolio)

        # get total risk over time
        total_static_risk = self.calcTotalRisk(init, 0, portfolio)
        total_continuous_risk = self.calcTotalRisk(init, contrib, portfolio)
        
        # get total returns over time
        total_static_returns = self.calcTotalReturns(init, 0, weights, returns)
        total_continuous_returns = self.calcTotalReturns(init, contrib, weights, returns)
        
        self.displayGraph(total_static_risk, total_static_returns, total_continuous_risk, total_continuous_returns, inflation, returns.index)
        
        
    def getClosingPrices(self, tickers, start_date, end_date):
        """Get the closing prices of tickers as dataframe (remove null rows)"""
        portfolio = pd.DataFrame()
        for ticker in tickers:
            portfolio[ticker] = pdr.get_data_yahoo(ticker, start_date, end_date, interval="m").Close # change interval here when changing INTERVAL
        return portfolio.dropna()
        
    
    def getReturns(self, prices_df):
        """Calculate the returns of the price dataframe"""
        shift = np.array(prices_df)[1:, :] - np.array(prices_df)[:-1,:]
        returns = pd.DataFrame(np.divide(shift, np.array(prices_df)[:-1,:]) + 1, columns = prices_df.columns)
        returns.index = prices_df.index[:-1]
        return returns
    

    def calcTotalRisk(self, init, contrib, portfolio):
        """Get list of risk over time"""
        return [ init + contrib * i for i in range(portfolio.shape[0]-1) ]

    
    def calcTotalReturns(self, init, contrib, weights, returns_df):
        returns = {}
        for i in range(returns_df.shape[1]):
            ticker = returns_df.columns[i]
            t_returns = returns_df[ticker]
            returns[ticker] = ( self.calcReturns(init*weights[i], contrib*weights[i], t_returns) )
        return np.array(pd.DataFrame(returns)).sum(axis=1)
    
    
    def calcReturns(self, init, contrib, pct_changes):
        """Get list of returns over time"""
        returns = []
        for i in range(len(pct_changes)):
            returns.append( ( returns[i-1] + contrib ) * pct_changes[i] if i > 0 else init * pct_changes[0] )
        return returns

    
    def displayGraph(self, static_risk, static_returns, continuous_risk, continuous_returns, inflation, xticks):
        fig = Figure(figsize=(5,4))
        fig.set_figheight(10)
        fig.set_figwidth(20)
        a = fig.add_subplot(111)
        if self.use_static.get():
            a.plot(xticks, static_risk, "y--")
            a.plot(xticks, static_returns, "y-")
        if self.use_cont.get():
            a.plot(xticks, continuous_risk, "b--")
            a.plot(xticks, continuous_returns, "b-")
        if self.use_inf:
            a.plot(xticks, inflation, "r-")
        a.set_title("Risk vs Returns of Portfolio")
        a.set_ylabel("$")
        a.set_xlabel("Time")
        plt.setp(a.xaxis.get_majorticklabels(), rotation=90)

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().grid(row=7)
        canvas.draw()
        
        
    def displayError(self, msg):
        """Output error message to screen"""
        self.error.configure(text=msg)
        
        
    def clearError(self):
        """Clear error message"""
        self.error.configure(text="")
        
    
    def getInflation(self):
        """Get inflation rate from CSV file"""
        cpi = pd.read_csv(CPI_CSV)
        cpi[CPI_COLUMN] = cpi[CPI_COLUMN].astype(float)
        inflation = pd.DataFrame( ( np.array(cpi[CPI_COLUMN])[1:] - np.array(cpi[CPI_COLUMN])[:-1] ) / np.array(cpi[CPI_COLUMN])[:-1] + 1, columns=["INFLATION_RATE"])
        inflation.index = [ pd.Timestamp(str(d)) for d in cpi.DATE[:-1] ] 
        return inflation

        
        
root = tk.Tk()
app = Application(master=root)
app.mainloop()


# In[ ]:




