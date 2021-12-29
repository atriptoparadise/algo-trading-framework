import numpy as np
import ta
import pandas as pd
import yfinance as yf
from strategy import RSIDivergence


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
        trades = 0
        print('')
        print('New Test')
        print('-' * 80)


        for row in range(rLimit):
            if buy[row] == 0 or row <= latest_row:
                continue
            
            buy_price = self.ohlc.loc[row, 'open']
            cur_qty = cur_amount / buy_price 
            print(' ')
            print(f"Buy at ${buy_price} - {self.ohlc.loc[row, 'timestamps']}")
            
            # Find sell for each buy
            for i in range(row + 1, rLimit):
                latest_row = i
                # Pass when no sell signal and beyond stop loss
                if sell[i] == 0 and self.ohlc.loc[i, 'low'] > buy_price * (1 - stopLoss):
                    continue
                
                # Stop Loss
                if self.ohlc.loc[i, 'low'] <= buy_price * (1 - stopLoss):
                    # If open with stop loss then sell at open price, otherwise sell at stop loss
                    sell_price = np.min([self.ohlc.loc[i, 'open'], buy_price * (1 - stopLoss)])
                    cur_amount = cur_qty * sell_price
                    print(f"Sell at ${sell_price} - {self.ohlc.loc[i, 'timestamps']} - stop loss")
                    print(f"--- Equity: {cur_amount}")
                    trades += 1
                    break
                
                # Sell by bear or RSI level crossover 
                sell_price = self.ohlc.loc[i, 'open']
                cur_amount = cur_qty * sell_price
                print(f"Sell at ${sell_price} - {self.ohlc.loc[i, 'timestamps']}")
                print(f"--- Equity: {cur_amount}")
                trades += 1
                break
        
        print(' ')
        print('-' * 80)
        print(f'Total trades: {trades}')
        print(f'Buy and Hold return: {round((self.ohlc.close.values[-1] - self.ohlc.open.values[0]) * 100 / self.ohlc.open.values[0], 2)}%')
        print(f'Strategy return: {round((cur_amount - initial_amount)* 100 / initial_amount, 2)}%')
        print(' ')


if __name__ == '__main__':
    # prepare data
    start = "2020-1-3"
    end = "2021-12-26"
    ticker = 'VIXY'
    yfObj = yf.Ticker(ticker)
    data = yfObj.history(start=start, end=end, interval='1h').reset_index()
    data.columns = [i.lower() for i in data.columns.values]
    data.rename({'index': 'timestamps'}, axis=1, inplace=True)

    # df = pd.read_csv('Binance_ETHUSDT_minute (1).csv')
    # df = df.iloc[::-1]
    # df = df.reset_index()
    # df = df.drop(columns=['unix', 'index', 'symbol', 'tradecount'])
    # df.date = pd.DatetimeIndex(df.date)

    # data = df.set_index('date').resample('10T').agg({'open': 'first',
    #                                                     'high': 'max',
    #                                                     'low': 'min',
    #                                                     'close': 'last'})
    # data = data.reset_index()
    # data.rename({'date': 'timestamps'}, axis=1, inplace=True)


    # data = pd.read_csv('test_ETH.csv')


    # Get signals from strategy
    strat = RSIDivergence(data)
    strat.getSignals(period=18, stopLoss=0.1, pivotLookBackLeft=1, pivotLookBackRight=2, 
                    rangeMin=5, rangeMax=60, RSISell=80, 
                    bullSignal=True, bullHiddenSignal=True, bearSignal=True, sellRSISignal=True)
    res = strat.ohlc

    # BackTest
    test = BackTest(res)
    test.backTest()
