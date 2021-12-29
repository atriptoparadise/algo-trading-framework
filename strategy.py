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
import numpy as np
import ta
import pandas as pd
import yfinance as yf
from datetime import datetime


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

	def pivotHigh(self, positionId, pivotLookBackLeft=3, pivotLookBackRight=1):
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
					buySigs.append(lookId)
			i += 1
		return self.ohlc.loc[buySigs, 'timestamps']
		# bull hidden


class RSI:
	'''
	init with ohlc data of pandas dataframe : 
	best to subset data to only required candles;

	output:
	'''
	def __init__(self, ohlc):
		self.ohlc = ohlc

		# ohlc format matches
		c = set(ohlc.columns.values)
		columnNames = {'timestamps', 'open', 'high', 'low', 'close'}
		assert (len(c.intersection(columnNames)) ==5), "pass ohlc parameter as pandas Dataframe of columns ['timestamps', 'open', 'high', 'low', 'close']"
		self.ohlc.sort_values('timestamps', inplace=True, ignore_index=True)
		
		self.ohlc['buy'] = 0
		self.ohlc['sell'] = 0

	def getHighs(self, data: np.array, pivotLookBackLeft=1, pivotLookBackRight=2):
		'''
		Finds highs in an array that satisfies left and right range condition.

		return: list of index.
		'''
		n = len(data)
		high = []
		for idx in range(pivotLookBackLeft, n - pivotLookBackRight):
			if data[idx] > np.max(data[idx - pivotLookBackLeft:idx]) and data[idx] > np.max(data[idx + 1:idx + pivotLookBackRight + 1]):
				high.append(idx)
		return high

	def getLows(self, data: np.array, pivotLookBackLeft=1, pivotLookBackRight=2):
		'''
		Finds lows in an array that satisfies left and right range condition.

		return: list of index.
		'''
		n = len(data)
		low = []
		for idx in range(pivotLookBackLeft, n-pivotLookBackRight):
			if data[idx] < np.min(data[idx - pivotLookBackLeft:idx]) and data[idx] < np.min(data[idx + 1:idx + pivotLookBackRight + 1]):
				low.append(idx)
		return low

	def getHigher(self, high_idx, highs, rangeMin=5, rangeMax=60):
		'''
		Finds Higher in an array of highs or lows.
		
		return: list of index.
		'''
		higher = []
		for i, idx in enumerate(high_idx):
			if i == 0:
				continue
			if highs[idx] > highs[high_idx[i - 1]] and rangeMin <= (idx - high_idx[i - 1]) <= rangeMax:
				higher.append(idx)
		return higher

	def getLower(self, low_idx, lows, rangeMin=5, rangeMax=60):
		'''
		Finds Higher in an array of highs or lows.
		
		return: list of index.
		'''
		lower = []
		for i, idx in enumerate(low_idx):
			if i == 0:
				continue
			if lows[idx] < lows[low_idx[i - 1]] and rangeMin <= (idx - low_idx[i - 1]) <= rangeMax:
				lower.append(idx)
		return lower

	def getHigherHighs(self, data: np.array, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60):
		'''
		Finds consecutive higher highs in an array.

		return: list of index of higher highs and index of highs.
		'''
		# Get highs
		high_idx = self.getHighs(data, 
							pivotLookBackLeft=pivotLookBackLeft, 
							pivotLookBackRight=pivotLookBackRight)
		highs = data[high_idx]
		return self.getHigher(high_idx, highs, rangeMin, rangeMax), high_idx

	def getHigherLows(self, data: np.array, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60):
		'''
		Finds consecutive higher lows in an array.

		return: list of index of higher lows and index of lows.
		'''
		low_idx = self.getLows(data, 
							pivotLookBackLeft=pivotLookBackLeft, 
							pivotLookBackRight=pivotLookBackRight)
		lows = data[low_idx]
		return self.getHigher(low_idx, lows, rangeMin, rangeMax), low_idx

	def getLowerLows(self, data: np.array, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60):
		'''
		Finds consecutive lower lows in an array.

		return: list of index of lower lows and index of lows.
		'''
		low_idx = self.getLows(data, 
							pivotLookBackLeft=pivotLookBackLeft, 
							pivotLookBackRight=pivotLookBackRight)
		lows = data[low_idx]
		return self.getLower(low_idx, lows, rangeMin, rangeMax), low_idx

	def getLowerHighs(self, data: np.array, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60):
		'''
		Finds consecutive lower highs in an array.

		return: list of index of lower highs and index of highs.
		'''
		high_idx = self.getHighs(data, 
							pivotLookBackLeft=pivotLookBackLeft, 
							pivotLookBackRight=pivotLookBackRight)
		highs = data[high_idx]
		return self.getLower(high_idx, highs, rangeMin, rangeMax), high_idx

	def getBullRegular(self, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60):
		oscHL, low_idx = self.getHigherLows(self.ohlc.osc, pivotLookBackLeft, pivotLookBackRight, rangeMin, rangeMax)
		
		priceLL = []
		price = self.ohlc.low.values

		for i in range(1, len(low_idx)):
			if price[low_idx[i]] < price[low_idx[i - 1]] and 60 > low_idx[i] - low_idx[i - 1] > 5:
				priceLL.append(low_idx[i])

		bull = list(set(priceLL) & set(oscHL))
		
		if not bull:
			return
		bull = [i + pivotLookBackRight + 1 for i in bull if i + pivotLookBackRight + 1 < self.ohlc.shape[0]]
		self.ohlc.loc[bull, 'buy'] = 1
		return self.ohlc[self.ohlc.index.isin(bull)]

	def getBullHidden(self, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60):
		oscLL, low_idx = self.getLowerLows(self.ohlc.osc, pivotLookBackLeft, pivotLookBackRight, rangeMin, rangeMax)
		
		priceHL = []
		price = self.ohlc.low.values

		for i in range(1, len(low_idx)):
			if price[low_idx[i]] > price[low_idx[i - 1]] and 60 > low_idx[i] - low_idx[i - 1] > 5:
				priceHL.append(low_idx[i])

		bullHidden = list(set(priceHL) & set(oscLL))
		if not bullHidden:
			return
		bullHidden = [i + pivotLookBackRight + 1 for i in bullHidden if i + pivotLookBackRight + 1 < self.ohlc.shape[0]]
		self.ohlc.loc[bullHidden, 'buy'] = 1
		return self.ohlc[self.ohlc.index.isin(bullHidden)]

	def getBearRegular(self, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60):
		oscLH, high_idx = self.getLowerHighs(self.ohlc.osc, pivotLookBackLeft, pivotLookBackRight, rangeMin, rangeMax)	
		
		priceHH = []
		price = self.ohlc.high.values

		for i in range(1, len(high_idx)):
			if price[high_idx[i]] > price[high_idx[i - 1]] and 60 > high_idx[i] - high_idx[i - 1] > 5:
				priceHH.append(high_idx[i])
		
		bear = list(set(priceHH) & set(oscLH))
		if not bear:
			return 
		bear = [i + pivotLookBackRight + 1 for i in bear if i + pivotLookBackRight + 1 < self.ohlc.shape[0]]
		self.ohlc.loc[bear, 'sell'] = 1
		return self.ohlc[self.ohlc.index.isin(bear)]

	def getSellRSI(self, RSISell=80):
		self.ohlc.loc[self.ohlc[self.ohlc.osc >= RSISell].index, 'sell'] = 1
		return self.ohlc[self.ohlc.osc >= RSISell]
	
	def getSignals(self, period=18, stopLoss=0.1, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60, RSISell=80):
		osc = ta.momentum.RSIIndicator(self.ohlc.close, period).rsi()
		self.ohlc['osc'] = osc

		bull = self.getBullRegular(pivotLookBackLeft, pivotLookBackRight, rangeMin, rangeMax)
		bullHidden = self.getBullHidden(pivotLookBackLeft, pivotLookBackRight, rangeMin, rangeMax)
		bear = self.getBearRegular(pivotLookBackLeft, pivotLookBackRight, rangeMin, rangeMax)
		sellRSI = self.getSellRSI(RSISell)
		return bull, bullHidden, bear, sellRSI



if __name__ == '__main__':
	# prepare data
	start = "2020-1-01"
	end = "2021-12-28"
	ticker = 'NVDA'
	yfObj = yf.Ticker(ticker)
	data = yfObj.history(start=start, end=end, interval='1h').reset_index()
	data.columns = [i.lower() for i in data.columns.values]
	data.rename({'index': 'timestamps'}, axis=1, inplace=True)
	
	strat = RSI(data)
	bull, bullHidden, bear, sellRSI = strat.getSignals(period=18, stopLoss=0.1, pivotLookBackLeft=1, pivotLookBackRight=2, rangeMin=5, rangeMax=60, RSISell=80)
	print(strat.ohlc[strat.ohlc.buy==1])