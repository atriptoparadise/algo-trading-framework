# hiddenBullCond = plotHiddenBull and priceHL and oscLL and plFound
# osc = rsi(src, len)
# len = input(title="RSI Period", minval=1, defval=9)
# src = input(title="RSI Source", defval=close)
# lbR = input(title="Pivot Lookback Right", defval=3)
# lbL = input(title="Pivot Lookback Left", defval=1)
# takeProfitRSILevel = input(title="Take Profit at RSI Level", minval=70, defval=80)

# // Price: Higher Low
# priceHL = low[lbR] > valuewhen(plFound, low[lbR], 1)

# // Osc: Lower Low
# oscLL = osc[lbR] < valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])

# // plFound pivot lows: number of bars with higher lows  
# Pivot Point Highs are calculated by the number of bars with lower highs on either side of a Pivot Point High calculation.
# plFound = na(pivotlow(osc, lbL, lbR)) ? false : true
import numpy
import ta

class Strategy:
	'''
	init with ohlc data of pandas dataframe : 
	best to subset data to only required candles;

	output: buy indicators of 1, sell indicators of -1;
	'''
	def __init__(self, ohlc):
		self.ohlc = ohlc

		# ohlc format matches
		c = set(ohlc.columns.values)
		columnNames = {'timestamps', 'open', 'high', 'low', 'close'}
		assert (len(c.intersection(columnNames)) ==5), "pass ohlc parameter as pandas Dataframe of columns ['timestamps', 'open', 'high', 'low', 'close']"
		self.ohlc.sort_values('timestamps', inplace=True, ignore_index=True)

	def pivotLow(self, positionId, pivotLookBackLeft=3, pivotLookBackRight=1):
		'''
		PivotLows are calculated by the number of bars with higher lows on either side of a Pivot Point Low calculation.
		'''
		n = len(self.ohlc)
		
		lows = self.ohlc.loc[positionId - pivotLookBackLeft : positionId + pivotLookBackRight + 1, 'low']
		v = self.ohlc.loc[positionId, 'low']
		c = lows[lows > v]
		return c.index

	def pivotHigh(self):
		'''
		Pivot Point Highs are calculated by the number of bars 
		with lower highs on either side of a Pivot Point High calculation. 
		returns the index of bars in [left, right] range that fits the criteria
		'''
		n = len(self.ohlc)

		highs = self.ohlc.loc[n - (pivotLookBackRight + pivotLookBackLeft) :, 'high']
		v = self.ohlc.loc[n - (pivotLookBackRight+1), 'high']
		c = highs[highs < v]
		return c.index

	def rsiIndicator(self, RSIperiod=14, pivotLookBackLeft=1, pivotLookBackRight=3):
		'''
		pivotLookBackLeft determines the number of bars before, default to 3
		pivotLookBackRight determines the number of bars in the future, i.e. slippage, default to 1
		'''
		osc = ta.momentum.RSIIndicator(self.ohlc.close, RSIperiod).rsi()
		self.ohlc['osc'] = osc
		n = len(self.ohlc)
		lookId = n - pivotLookBackRight - 1
		# bull regular
		# // Osc: Higher Low
		# oscHL = osc[lbR] > valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])

		# // Price: Lower Low
		# priceLL = low[lbR] < valuewhen(plFound, low[lbR], 1)

		# bullCond = plotBull and priceLL and oscHL and plFound
		plows = self.pivotLow(lookId, pivotLookBackLeft, pivotLookBackRight)

		if len(plows) > 0:

			nowlow = self.ohlc.loc[lookId, 'low']
			priceLL = self.ohlc.loc[plows, 'low']
			# find the 2nd occurence when price higher lows happens
			priceLL = priceLL[priceLL > nowlow]

			nowOsc = osc[lookId]
			oscHL = osc[plows]
			oscHL = oscHL[oscHL < nowOsc]

			if len(priceLL) >= 1 and len(oscHL) >= 1:
				return 1

	def rsiIndicatorBackTester(self, RSIperiod=14, pivotLookBackLeft=3, pivotLookBackRight=1):
		'''
		pivotLookBackLeft determines the number of bars before, default to 3
		pivotLookBackRight determines the number of bars in the future, i.e. slippage, default to 1
		'''
		osc = ta.momentum.RSIIndicator(self.ohlc.close, RSIperiod).rsi()
		self.ohlc['osc'] = osc
		n = len(self.ohlc)
		
		i = 0
		iLimit = n-(RSIperiod + pivotLookBackLeft)
		buySigs = []
		while i < iLimit:
			lookId = n - pivotLookBackRight - 1 - i
			# bull regular
			# // Osc: Higher Low
			# oscHL = osc[lbR] > valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])

			# // Price: Lower Low
			# priceLL = low[lbR] < valuewhen(plFound, low[lbR], 1)

			# bullCond = plotBull and priceLL and oscHL and plFound
			plows = self.pivotLow(pivotLookBackLeft, pivotLookBackRight)

			if len(plows) > 0:

				nowlow = self.ohlc.loc[lookId, 'low']
				priceLL = self.ohlc.loc[plows, 'low']
				# find the 2nd occurence when price higher lows happens
				priceLL = priceLL[priceLL > nowlow]

				nowOsc = osc[lookId]
				oscHL = osc[plows]
				oscHL = oscHL[oscHL < nowOsc]

				if len(priceLL) >= 1 and len(oscHL) >= 1:
					buySigs.append(lookId)
			i += 1
		return self.ohlc.loc[buySigs, 'timestamps']
		# bull hidden



if __name__ == '__main__':
	import numpy as np
	import pandas as pd
	import yfinance as yf

	# prepare data
	start = '2021-10-01'
	end = '2021-12-26'
	ticker = 'NVDA'
	yfObj = yf.Ticker(ticker)
	data = yfObj.history(start=start, end=end).reset_index()
	data.columns = [i.lower() for i in data.columns.values]
	data.rename({'date': 'timestamps'}, axis=1, inplace=True)
	
	strat = Strategy(data)
	print(strat.rsiIndicatorBackTester())