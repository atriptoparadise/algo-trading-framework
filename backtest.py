import numpy as np
import ta
import pandas as pd
import yfinance as yf
from strategy import RSI


class BackTest:
    '''
	Input: dataframe with timestamps, ohlc, buy, and sell.

	Output:
	'''
    def __init__(self, ohlc):
        self.ohlc = ohlc

        # ohlc format matches
        c = set(ohlc.columns.values)
        columnNames = {'timestamps', 'open', 'high', 'low', 'close', 'buy', 'sell'}
        assert (len(c.intersection(columnNames)) ==7), "pass ohlc parameter as pandas Dataframe of columns ['timestamps', 'open', 'high', 'low', 'close', 'buy', 'sell']"
        self.ohlc.sort_values('timestamps', inplace=True, ignore_index=True)

    def backTest(self, stopLoss=0.1, initial_amount=100000, order_percent=100):
        buy = self.ohlc.buy.values
        sell = self.ohlc.sell.values
        rLimit = self.ohlc.shape[0]
        latest_row = 0
        cur_amount = initial_amount
        cur_qty = 0

        for row in range(rLimit):
            if buy[row] == 0 or row <= latest_row:
                continue
            
            buy_price = self.ohlc.loc[row, 'open']
            cur_qty = cur_amount / buy_price 
            print(f"Buy at ${buy_price} - {self.ohlc.loc[row, 'timestamps']}")
            
            
            for i in range(row + 1, rLimit):
                latest_row = i

                if sell[i] == 0 and self.ohlc.loc[i, 'low'] > buy_price * (1 - stopLoss):
                    continue
                
                # Stop Loss
                if self.ohlc.loc[i, 'low'] <= buy_price * (1 - stopLoss):
                    # If open with stop loss then sell at open price, otherwise sell at stop loss
                    sell_price = np.min([self.ohlc.loc[i, 'open'], buy_price * (1 - stopLoss)])
                    cur_amount = cur_qty * sell_price
                    print(f"Sell at ${sell_price} - {self.ohlc.loc[i, 'timestamps']}")
                    print(f"--- Equity: {cur_amount}")
                    break
                
                # Sell by bear or RSI level crossover 
                sell_price = self.ohlc.loc[i, 'open']
                cur_amount = cur_qty * sell_price
                print(f"Sell at ${sell_price} - {self.ohlc.loc[i, 'timestamps']}")
                print(f"--- Equity: {cur_amount}")
                break


if __name__ == '__main__':
    # prepare data
    start = "2020-1-16"
    end = "2021-12-26"
    ticker = 'NVDA'
    yfObj = yf.Ticker(ticker)
    data = yfObj.history(start=start, end=end, interval='1h').reset_index()
    data.columns = [i.lower() for i in data.columns.values]
    data.rename({'index': 'timestamps'}, axis=1, inplace=True)

    # Get signals from strategy
    strat = RSI(data)
    strat.getSignals(period=18, stopLoss=0.1, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60, RSISell=80)
    res = strat.ohlc

    # BackTest
    test = BackTest(res)
    test.backTest()
