import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import stats
import numpy as np

def matplotlib_config():
		mpl.rc('text', color='#3ba6d8')
		mpl.rc('axes', facecolor='none',edgecolor="#3ba6d8",labelcolor="#3ba6d8",titlesize=15,labelsize=10,grid=True)
		mpl.rc('lines', linewidth=1,color="#3ba6d8")
		mpl.rc('grid',color='#f2f2f2',alpha=0.3,linewidth=1)
		mpl.rc('xtick',color='#3ba6d8')
		mpl.rc('ytick',color='#3ba6d8')

def basic_graph(x,title=''):
	matplotlib_config()
	plt.figure()
	plt.plot(x,color='red')
	plt.legend(handles=stat_patch_list(x),fontsize=10,loc='upper right')
	#plt.ylim(-0.5,1.0)
	plt.savefig(title +'.png',transparent=True)
	plt.close()

def time_series_graph(x,title=''):
	matplotlib_config()
	plt.figure()
	plt.plot(x,color='red')
	plt.savefig(title +'.png',transparent=True)
	plt.close()

def graph_with_subsampling(x,subsample_list,title=''):
	matplotlib_config()
	plt.figure()
	plt.plot(x,color='red')
	for sample in subsample_list:
		plt.plot(sample,color='blue')
	plt.legend(handles=stat_patch_list(x),fontsize=10,loc='upper right')
	plt.savefig(title +'.png',transparent=True)
	plt.close()

def prediction_graph(pred,y,title=''):
	matplotlib_config()
	plt.figure()
	plt.plot(y,color='blue')
	plt.plot(pred,color='red')
	plt.title(title)
	#plt.xlabel('data in series')
	#plt.ylabel('std21')
	plt.legend(handles=comparing_stat_patch_list(pred,y),fontsize=10,loc='upper right')
	plt.savefig(title +'.png',transparent=True)
	plt.close()

def histogram_graph(y,title=''):
	matplotlib_config()
	plt.figure()
	plt.hist(y,bins=100,color='blue')
	plt.title(title)
	plt.legend(handles=stat_patch_list(y),fontsize=10,loc='upper right')
	plt.savefig(title +'.png',transparent=True)
	plt.close()

def comparing_stat_patch_list(pred,y):
	patch1 = mpatches.Patch(color='red', label= 'mean:'+ ('%03.6f' % np.mean(pred)))
	patch2 = mpatches.Patch(color='red', label= 'std:'+ ('%03.6f' % np.std(pred)))
	patch3 = mpatches.Patch(color='red', label= 'skewness:'+ ('%03.3f' % stats.skewness(pred)))
	patch4 = mpatches.Patch(color='red', label= 'kurtosis:'+ ('%03.3f' % stats.kurtosis(pred)))
	patch5 = mpatches.Patch(color='blue', label= 'mean:'+ ('%03.6f' % np.mean(y)))
	patch6 = mpatches.Patch(color='blue', label= 'std:'+ ('%03.6f' % np.std(y)))
	patch7 = mpatches.Patch(color='blue', label= 'skewness:'+ ('%03.3f' % stats.skewness(y)))
	patch8 = mpatches.Patch(color='blue', label= 'kurtosis:'+ ('%03.3f' % stats.kurtosis(y)))
	#patch9 = mpatches.Patch(color='black', label= 'MAPE:'+ ('%03.3f' % stats.mape(pred, y)))
	patch10 = mpatches.Patch(color='black', label= 'RMSE:'+ ('%03.6f' % stats.rmse(pred, y)))
	#plt.text(.25,.5,str(np.mean(pred)))
	return [patch5,patch6,patch7,patch8,patch1,patch2,patch3,patch4,patch10]

def stat_patch_list(x):
	patch0 = mpatches.Patch(label= 'data num:'+ str((x.size)))
	patch1 = mpatches.Patch(label= 'mean:'+ ('%03.6f' % np.mean(x)))
	patch2 = mpatches.Patch(label= 'std:'+ ('%03.6f' % np.std(x)))
	patch3 = mpatches.Patch(label= 'skewness:'+ ('%03.3f' % stats.skewness(x)))
	patch4 = mpatches.Patch(label= 'kurtosis:'+ ('%03.3f' % stats.kurtosis(x)))
	return [patch0,patch1,patch2,patch3,patch4]

def basic_patch_list():
	patch1 = mpatches.Patch(color='red', label= 'prediciton')
	patch2 = mpatches.Patch(color='blue', label= 'actual value')
	return [patch1,patch2]

def patch_list(pred,y):
	return stat_patch_list(pred, y)
