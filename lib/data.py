import pandas as pd
import numpy as np
#import arch
import os
import stats
import visualizer
from datetime import datetime
import warnings
#import talib as ta

warnings.filterwarnings('ignore')

class data_manager():
	def __init__(self):
		self.csv_file = ''
		pass

#where to normalize?
	def prepare_pd(self):
		self.read_csv()
		self.parse_date()
		self.data_begin_date = None
		self.data_end_date = None
		self.add_columns()

	def add_columns(self,is_financial = True, contain_adjusted = True,contain_volume=True,validation_split=0.81):
		if is_financial:
			if contain_adjusted:
				self.data['Log Return'] = np.log(self.data['Adj Close']) - np.log(self.data['Adj Close'].shift(1))
				self.data['Pct Change'] = self.data['Adj Close'].pct_change().dropna()
			else:
				self.data['Log Return'] = np.log(self.data['Close']) - np.log(self.data['Close'].shift(1))
				self.data['Pct Change'] = self.data['Close'].pct_change().dropna()
			# self.data['U'] = np.log(self.data['High']/self.data['Open'])
			# self.data['D'] = np.log(self.data['Low']/self.data['Open'])
			# self.data['C'] = np.log(self.data['Close']/self.data['Open'])
			# self.data['A'] = self.data.apply(lambda x: 1 if x['Log Return'] >=0 else 0 ,axis=1)
			# self.data['Y'] = self.data['A'].shift(-1)
			# self.data['Daily Volatility'] = np.sqrt(0.511*((self.data['U']-self.data['D'])**2) - 0.019*(self.data['C']*(self.data['U']+self.data['D'])-2*self.data['U']*self.data['D']) - 0.383*(self.data['C']**2))
			# self.data['Simple 10-day MA'] = pd.rolling_mean(self.data['Close'],10)
			# self.data['Weighted 10-day MA'] = ta.WMA(np.array(self.data['Close'],dtype=np.float),timeperiod=10)
			# self.data['Momentum'] = ta.MOM(np.array(self.data['Close'],dtype=np.float), timeperiod=10)
			# self.data['Stochastic K'] = stats.stochastic_K(self.data['Close'], self.data['High'], self.data['Low'], 14)
			# self.data['Stochastic D'] = stats.stochastic_D(self.data['Close'], self.data['High'], self.data['Low'], 14,3)
			# self.data['sign'] = self.data.apply(lambda x: 1 if x['Log Return'] >=0 else -1 ,axis=1)
			# self.data['RSI'] = ta.RSI(np.array(self.data['Close'],dtype=np.float), timeperiod=14)
			# self.data['MACD'] = ta.MACD(np.array(self.data['Close'],dtype=np.float),fastperiod=12, slowperiod=26, signalperiod=9)[0]
			# self.data['Larry William R'] = ta.WILLR(np.array(self.data['High'],dtype=np.float), np.array(self.data['Low'],dtype=np.float),np.array(self.data['Close'],dtype=np.float), timeperiod=14)
			# self.data['AD'] = stats.AD(self.data['Close'],self.data['High'],self.data['Low'])
			# self.data['CCI'] = ta.CCI(np.array(self.data['Close'],dtype=np.float), np.array(self.data['Low'],dtype=np.float), np.array(self.data['High'],dtype=np.float), timeperiod=20)
			# if contain_volume:
			# 	self.data['Mean Volume 300'] = pd.rolling_mean(self.data['Volume'], 300)
			# 	self.data['Std Volume 300'] = pd.rolling_std(self.data['Volume'], 300)
			# 	self.data['Normalized Volume 300'] = (self.data['Volume']-self.data['Mean Volume 300'])/self.data['Std Volume 300']
				#self.data['Mean Volume 1000'] = pd.rolling_mean(self.data['Volume'], 1000)
				#self.data['Std Volume 1000'] = pd.rolling_std(self.data['Volume'], 1000)
				#self.data['Normalized Volume 1000'] = (self.data['Volume']-self.data['Mean Volume 1000'])/self.data['Std Volume 1000']
			# self.data['Stdev2'] = pd.rolling_std(self.data['Log Return'], 2)
			# self.data['Stdev7'] = pd.rolling_std(self.data['Log Return'], 7)
			# self.data['Stdev14'] = pd.rolling_std(self.data['Log Return'], 14)
			# self.data['Stdev21'] = pd.rolling_std(self.data['Log Return'], 21)
			# self.data['Stdev28'] = pd.rolling_std(self.data['Log Return'], 28)
			#GARCH fitting
			self.data = self.data.dropna()
			#
			# self.data['Forecast Volatility'] = np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]'])
			# self.data['Annualized Forecast Volatility'] = self.data['Forecast Volatility']*np.sqrt(252)

	def fill_garch(self,ret,validation_split=0.2):
		c = 100
		am = arch.arch_model(ret[:int((1-validation_split)*ret.size)]*c)
		res = am.fit(update_freq=5,disp='on')
		new_cond_vol = np.zeros(ret.size)
		c_vol = res.conditional_volatility
		for i in range(ret.size):
			if c_vol.size > i:
				new_cond_vol[i] = c_vol[i]
			else:
				new_cond_vol[i] = np.sqrt(res.params['omega'] + res.params['alpha[1]'] * ((ret[i-1]*c)-res.params['mu'])**2 + (new_cond_vol[i - 1]**2)*res.params['beta[1]'])
		return new_cond_vol/c
			#return None
	def read_csv(self):
		self.data = pd.read_csv(self.csv_file)

	def data_stat(self):
		pass

	def get_pd_table(self,**kwargs):
		table = self.data
		if 'date_start' in kwargs:
			table = table.ix[table['Date'] > kwargs['date_start']]
			kwargs.pop('date_start')
		if 'date_end' in kwargs:
			table = table.ix[self.data['Date'] < kwargs['date_end']]
			kwargs.pop('date_end')

		return table

	def get_dataset(self,column_list,look_back = 10,normalize_width = 0,normalize_scheme = None,train_ratio = 0.8):
		#get specified pd table
		specified_table = self.data(column_list)
		#Normalize
		if normalize_width:
			for e in normalize_scheme:
				pass
		#Look-Back

		#Divide data into train/test
		train_x = None
		train_y = None
		test_x = None
		test_y = None
		return self._create_dataset(train_x,train_y),self._create_dataset(test_x,test_y)
		pass

	def create_dataset(self,x_set, y_set,look_back=10):
		data_x, data_y = [], []
		for i in range(len(x_set)-look_back-1):
			a = x_set[i:(i+look_back)]
			data_x.append(a)
			data_y.append(y_set[i + look_back - 1])
		return np.array(data_x), np.array(data_y)

	def parse_date(self):
		self.data['Date'] = pd.to_datetime(self.data['Date'])
		oldtime_table = self.data.copy()
		oldtime_table['Date'] -= pd.Timedelta(100,'Y')
		self.data = self.data.where(self.data['Date'] < pd.to_datetime('2017/1/1'),oldtime_table)

class gdc(data_manager):
	def __init__(self):
		super().__init__()
		self.csv_file = ''

	def prepare_pd(self):
		self.data = pd.read_csv(self.csv_file)


class snp500(data_manager):
	def __init__(self):
		self.csv_file = './data/SNP500/snp500.csv'

	def get_code_list(self):
		t = pd.read_csv('./data/SNP500/SNP500_Individuals/constituents.csv')
		return t['Symbol'].as_matrix()


class snp500_individual(data_manager):
	def __init__(self,code):
		#os.chdir("./data/SNP500/SNP500_Individuals")
		self.csv_file = "./data/SNP500/SNP500_Individuals/" +  str(code) + '.csv'

	def read_csv(self):
		self.data = pd.read_csv(self.csv_file,dtype={'Open': np.float32,'Low': np.float32,'High': np.float32,'Close': np.float32,'Volume': np.float32,'Adj_Close': np.float32})
		self.data.rename(columns={'Adj_Close':'Adj Close'},inplace=True)
		#self.data = self.data.iloc[::-1]



class nikkei225_individual(data_manager):
	def __init__(self,code):
		#os.chdir("./data/Nikkei225/Nikkei225_Individuals")
		self.csv_file = "../data/Nikkei225/Nikkei225_Individuals/" + str(code) + '.csv'

	def read_csv(self):
		self.data = pd.read_csv(self.csv_file,names=['Date','Open','High','Low','Close','Volume','Adj Close'],dtype={'Open': np.float32,'Low': np.float32,'High': np.float32,'Close': np.float32,'Volume': np.float32,'Adj Close': np.float32})
		self.data = self.data.iloc[::-1]

class nikkei225(data_manager):
	def __init__(self):
		#os.chdir("./data/Nikkei225")
		self.csv_file = '../data/Nikkei225/nikkei 225.csv'

	def prepare_pd(self):
		self.read_csv()
		self.data_begin_date = None
		self.data_end_date = None
		self.add_columns(True,False,False)

	def read_csv(self):
		#self.data = pd.read_csv(self.csv_file,names=['Date','Open','Low','High','Close'],dtype={'Open': np.float32,'Low': np.float32,'High': np.float32,'Close': np.float32})
		self.data = pd.read_csv(self.csv_file)
		self.data = self.data.rename(index=str, columns={"date": "Date", "open": "Open","low": "Low","high": "High","close": "Close"})

	def get_code_list(self):
		t = pd.read_csv("../data/Nikkei225/Nikkei225_Individuals/nikkei225-stock-prices.csv")
		return t['SC'].as_matrix()

class djia(data_manager):
	def __init__(self):
		super().__init__()
		self.csv_file = ''


