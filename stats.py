import numpy as np
import pandas as pd

def auto_correlation(x,lag = 1):
	a = pd.Series(np.reshape(x,(-1)))
	b = a.autocorr(lag = lag)
	if np.isnan(b) or np.isinf(b):
		return 0
	return b

def acf(x,max_lag=1000):
	acf = []
	for i in range(max_lag):
		acf.append(auto_correlation(x,lag=i+1))
	return np.array(acf)

