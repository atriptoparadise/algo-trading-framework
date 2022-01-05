import numpy as np
import ta
import pandas as pd
import yfinance as yf
from strategy import RSIDivergence
from datetime import datetime




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

    def find_max_drawdown(self, price):
        mdd = 0
        peak = price[0]
        draw_list = []
        for x in range(len(price)):
            if price[x] > peak: 
                peak = price[x]
                if x < len(price) - 1:
                    draw_list = [x]
            dd = (peak - price[x]) / peak
            if dd > mdd:
                # print(dd, mdd, x)
                mdd = dd
                draw_list.append(x)
        return round(100 * mdd, 2)

    def backTest(self, stopLoss=0.1, initial_amount=100000, order_percent=100):
        buy = self.ohlc.buy.values
        sell = self.ohlc.sell.values
        rLimit = self.ohlc.shape[0]
        latest_row = 0
        cur_amount = initial_amount
        cur_qty = 0
        trades = 0
        buy_sigs, sell_sigs, sl_sigs = [], [], []
        trades_info = []
        print('')
        print('New Test')
        print('-' * 80)
        win = 0
        win_gross = 0
        lose = 0
        lose_gross = 0
        close_each_bar = []

        for row in range(rLimit):
            if row == 0:
                close_each_bar.append([self.ohlc.loc[row, 'timestamps'], 
                                    self.ohlc.loc[row, 'close'],
                                    cur_amount])
            if row <= latest_row:
                continue
            
            close_each_bar.append([self.ohlc.loc[row, 'timestamps'], 
                                    self.ohlc.loc[row, 'close'],
                                    cur_amount])

            if buy[row] == 0:
                continue
            buy_price = self.ohlc.loc[row, 'open']
            buy_time = self.ohlc.loc[row, 'timestamps']
            cur_qty = cur_amount / buy_price 
            
            print(' ')
            print(f"Buy  at ${round(buy_price, 2)} - {buy_time}")
            buy_sigs.append((row, trades))
            trades_info.append([trades + 1, 'Entry Long', buy_time, buy_price, '-'])

            # Find sell for each buy
            for i in range(row + 1, rLimit):
                latest_row = i
                # Pass when no sell signal and beyond stop loss
                if sell[i] == 0 and self.ohlc.loc[i, 'low'] > buy_price * (1 - stopLoss):
                    close_each_bar.append([self.ohlc.loc[i, 'timestamps'], 
                                            self.ohlc.loc[i, 'close'],
                                            cur_qty * self.ohlc.loc[i, 'close']])
                    continue
                
                # Stop Loss
                if self.ohlc.loc[i, 'low'] <= buy_price * (1 - stopLoss):
                    # If open with stop loss then sell at open price, otherwise sell at stop loss
                    sell_price = np.min([self.ohlc.loc[i, 'open'], buy_price * (1 - stopLoss)])
                    cur_amount = cur_qty * sell_price
                    sell_time = self.ohlc.loc[i, 'timestamps']
                    profit = round(100 * (sell_price - buy_price) / buy_price, 2)
                    print(f"Sell at ${round(sell_price, 2)} - {sell_time} - Profit: {profit}% (stop loss)")
                    print(f"--- Equity: {cur_amount}")
                    sl_sigs.append((i, trades))
                    
                    trades_info.append([trades + 1, 'Exit Long', sell_time, sell_price, profit])
                    trades += 1
                    lose += 1
                    lose_gross += (buy_price - sell_price) * cur_qty
                    close_each_bar.append([self.ohlc.loc[i, 'timestamps'], 
                                            self.ohlc.loc[i, 'close'],
                                            cur_amount])
                    break
                
                # Sell by bear or RSI level crossover 
                sell_price = self.ohlc.loc[i, 'open']
                cur_amount = cur_qty * sell_price
                sell_time = self.ohlc.loc[i, 'timestamps']
                profit = round(100 * (sell_price - buy_price) / buy_price, 2)
                print(f"Sell at ${round(sell_price, 2)} - {sell_time} - Profit: {profit}%")
                print(f"--- Equity: {cur_amount}")
                sell_sigs.append((i, trades))
                trades_info.append([trades + 1, 'Exit Long', sell_time, sell_price, profit])
                trades += 1

                if sell_price >= buy_price:
                    win += 1
                    win_gross += (sell_price - buy_price) * cur_qty
                else:
                    lose += 1
                    lose_gross += (buy_price - sell_price) * cur_qty

                close_each_bar.append([self.ohlc.loc[i, 'timestamps'], 
                                            self.ohlc.loc[i, 'close'],
                                            cur_amount])
                break
        
        print(' ')
        print('-' * 80)
        print(f"Start from {self.ohlc.loc[0, 'timestamps']} to {self.ohlc.loc[self.ohlc.shape[0] - 1, 'timestamps']}")
        print(' ')
        print(f'Total trades: {trades}')
        print(f'Buy and Hold return: {round((self.ohlc.close.values[-1] - self.ohlc.open.values[0]) * 100 / self.ohlc.open.values[0], 2)}%')
        print(f'Strategy return: {round((cur_amount - initial_amount)* 100 / initial_amount, 2)}%')
        print(f'Percent Profitbale: {round(win * 100 / (win + lose), 2)}%')
        print(f'Profit Factor: {round(win_gross / lose_gross, 2)}')
        print(' ')
        trades_df = pd.DataFrame(trades_info, columns=['trade #', 'type', 'timestamps', 'price', 'Profit %'])
        trades_df.to_csv('trades_df.csv', index=False)

        trades_trend = pd.DataFrame(close_each_bar, columns=['timestamps', 'close', 'strat_equity'])
        trades_trend['strat_equity'] = trades_trend['strat_equity'] * trades_trend['close'][0] / trades_trend['strat_equity'][0]
        trades_trend['close_return'] = trades_trend.close.pct_change(1)
        trades_trend['strat_equity_return'] = trades_trend.strat_equity.pct_change(1)
        trades_trend.to_csv('trades_trend.csv', index=False)

        # Only for 10 mins dataset
        sharpe_ratio_close = trades_trend['close_return'].mean() * ((252 * 8 * 6) ** .5) / trades_trend['close_return'].std() 
        sharpe_ratio_strat = trades_trend['strat_equity_return'].mean() * ((252 * 8 * 6) ** .5) / trades_trend['strat_equity_return'].std() 
        print(f'Buy and Hold Sharpe Ratio: {round(sharpe_ratio_close, 3)}')
        print(f'Strategy Sharpe Ratio: {round(sharpe_ratio_strat, 3)}')
        print(' ')
        
        print(f"Buy and Hold Max Drawdown: {self.find_max_drawdown(trades_trend['close'].values)}%")
        print(f"Strategy Max Drawdown: {self.find_max_drawdown(trades_trend['strat_equity'].values)}%")
        print(' ')
        return buy_sigs, sell_sigs, sl_sigs


if __name__ == '__main__':
    # # prepare data
    # start = "2020-1-3"
    # end = "2021-12-26"
    # ticker = 'TQQQ'
    # yfObj = yf.Ticker(ticker)
    # data = yfObj.history(start=start, end=end, interval='1h').reset_index()
    # data.columns = [i.lower() for i in data.columns.values]
    # data.rename({'index': 'timestamps'}, axis=1, inplace=True)

    # df = pd.read_csv('Binance_ETHUSDT_minute (1).csv')
    # df = df.iloc[::-1]
    # df = df.reset_index()
    # df = df.drop(columns=['unix', 'index', 'symbol', 'tradecount'])
    # df.date = pd.DatetimeIndex(df.date)

    # data = df.set_index('date').resample('10T').agg({'open': 'first',
    #                                                     'high': 'max',
    #                                                     'low': 'min',
    #                                                     'close': 'last'})
    # data = data.reset_index().dropna()
    # data.rename({'date': 'timestamps'}, axis=1, inplace=True)

    ticker = 'TQQQ'
    data = pd.read_csv('tqqq_10m.csv').dropna()
    data.timestamps = pd.DatetimeIndex(data.timestamps)
    data = data[data.timestamps >= datetime(2019, 12, 8)]


    # Get signals from strategy
    strat = RSIDivergence(data)
    strat.getSignals(period=18, stopLoss=0.1, pivotLookBackLeft=1, pivotLookBackRight=2, 
                    rangeMin=5, rangeMax=60, RSISell=80, 
                    bullSignal=True, bullHiddenSignal=True, bearSignal=True, sellRSISignal=True,
                    remove_first_250=True)
    res = strat.ohlc

    # BackTest
    test = BackTest(res)
    test.backTest()
