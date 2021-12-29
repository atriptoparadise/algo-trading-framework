import pandas as pd
import datetime as dt
import pandas_datareader.data as web
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from strategy import RSI
from backtest import BackTest


def candlestick_plot(stocks, ticker, signals):
  '''
  input: stocks - OHLC data; ticker - string; signals - three lists of IDX in stocks that give buy, sell, stoploss signals
  shows plotly interactive plot

  '''
  bs, ss, sls = signals
  minTime = data.timestamps.min()
  maxTime = data.timestamps.max()
  c_candlestick = go.Figure(data = [go.Candlestick(x = data.timestamps, 
                                               open = data['open'], 
                                               high = data['high'],
                                               low = data['low'],
                                               close = data['close'])])

  c_candlestick.update_xaxes(
    title_text = 'Date',
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
            dict(count = 1, label = '1M', step = 'month', stepmode = 'backward'),
            dict(count = 6, label = '6M', step = 'month', stepmode = 'backward'),
            dict(count = 1, label = 'YTD', step = 'year', stepmode = 'todate'),
            dict(count = 1, label = '1Y', step = 'year', stepmode = 'backward'),
            dict(step = 'all')])))

  for idx, tradeId in bs: #buy signals
      c_candlestick.add_annotation(x=data.loc[idx, 'timestamps'], y=data.loc[idx, 'low'],
                  text="buy",
                  showarrow=True,
                  arrowhead=1)
      
  for idx, tradeId in ss: #sell signals
      c_candlestick.add_annotation(x=data.loc[idx, 'timestamps'], y=data.loc[idx, 'high'],
                  text="sell",
                  showarrow=True,
                  arrowhead=1)
      
  for idx, tradeId in sls: #buy signals
      c_candlestick.add_annotation(x=data.loc[idx, 'timestamps'], y=data.loc[idx, 'high'],
                  text="stop",
                  showarrow=True,
                  arrowhead=1)

  c_candlestick.update_layout(
    title = {
        'text': '{} SHARE PRICE {} - {}'.format(ticker, minTime, maxTime),
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

  c_candlestick.update_yaxes(title_text = '{} Close Price', tickprefix = '$')
  c_candlestick.show()


if __name__ == '__main__':
  # prepare data
  start = "2020-1-01"
  end = "2021-12-28"
  ticker = 'NVDA'
  yfObj = yf.Ticker(ticker)
  data = yfObj.history(start=start, end=end, intervals='1h').reset_index()
  data.columns = [i.lower() for i in data.columns.values]
  data.rename({'index': 'timestamps', 'date':'timestamps'}, axis=1, inplace=True)
  
  strat = RSI(data)
  bull, bullHidden, bear, sellRSI = strat.getSignals(period=18, stopLoss=0.1, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60, RSISell=80)
  res = strat.ohlc
  print(strat.ohlc[strat.ohlc.buy==1])

  # BackTest
  test = BackTest(res)
  signals = test.backTest()
  
  candlestick_plot(data, ticker, signals)