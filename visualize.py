import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.autolayout']=True
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

def loss_history(date):
    def subsample(x):
        indices = [i*100 for i in range(x.size//100)]
        return x[indices]
    g_loss = np.load('./imgs/%s/g_loss.npy'%(date))
    d_loss = np.load('./imgs/%s/d_loss.npy'%(date))
    g_losses = np.load('./imgs/%s/g_losses.npy'%(date))
    d_losses = np.load('./imgs/%s/d_losses.npy'%(date))
    plt.figure(figsize=(16,10),dpi=600)
    ax = plt.subplot(111)
    plt.xlabel('batches',fontsize=36)
    plt.ylabel('loss',fontsize=36)
    g_colors = ['rosybrown','lightcoral','brown','firebrick','maroon']
    d_colors = ['darkslateblue','slateblue','mediumslateblue','royalblue','darkblue']
    names = ['feature_%i'%i for i in range (1,6)]
    patches = []
    patches.append(mpatches.Patch(color='red',label='Generator'))
    patches.append(mpatches.Patch(color='blue',label='Discriminator'))
    for c,n in zip(g_colors,names):
        patches.append(mpatches.Patch(color=c,label=n+'_generator'))
    for c,n in zip(d_colors,names):
        patches.append(mpatches.Patch(color=c,label=n+'_discriminator'))

    indices = [i*100 for i in range(g_loss.size//100)]
    #for i,c in zip(range(5),g_colors):
    #    ax.plot(indices,subsample(g_losses[:,i]),'-',color=c,linewidth=2,alpha=0.5)
    #for i,c in zip(range(5),d_colors):
    #    ax.plot(indices,subsample(d_losses[:,i]),'-',color=c,linewidth=2,alpha=0.5)
    ax.plot(indices,subsample(g_loss),'-',linewidth=4,color='red',alpha=0.8)
    ax.plot(indices,subsample(d_loss),'-',linewidth=4,color='blue',alpha=0.8)
    plt.yscale('log')
    plt.ylim(0.001,14)
    plt.xlim(0,10000)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True, ncol=12,handles=patches,fontsize=12)
    plt.savefig('%s_losses'%date,transparent=True)
    plt.close()



def time_series(x,file_name):
    plt.figure(dpi=150)
    plt.plot(x)
    plt.ylim(-0.25,0.25)
    plt.xlabel('Time Ticks($t$)',fontsize=20)
    plt.ylabel('Normalized Return',fontsize=20)
    plt.savefig(file_name+'.png',transparent=True)
    plt.close()


def leverage_effect(x,y,file_name):          
    plt.figure(dpi=150)
    plt.plot(x,y)
    plt.xlabel(r'$t$',fontsize=20)
    plt.ylabel(r'$L(t)$',fontsize=20)
    plt.axhline()
    plt.savefig(file_name+'.png',transparent=True)
    plt.close()

def distribution(x,y,file_name,scale='linear'):
    if scale is 'log':
        distribution_log(x, y, file_name)
    elif scale is 'linear':
        distribution_linear(x, y, file_name)

def distribution_linear(x,y,file_name):
    plt.figure(dpi=150)
    plt.plot(x,y,'.',markersize=8)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Normalized Scale in $\sigma$',fontsize=20)
    plt.ylabel('Probability Density Function $P(r)$',fontsize=40)
    plt.savefig(file_name+'_linear.png',transparent=True)
    plt.close()


def distribution_log(x,y,file_name):
    #split into positive and negative sides
    dist_x_pos = x[x > 0]
    dist_y_pos = y[-dist_x_pos.size:]
    dist_x_neg = -x[x < 0]
    dist_y_neg = y[:dist_x_neg.size]
    #for positive
    plt.figure(dpi=150)
    plt.plot(dist_x_pos,dist_y_pos,'.')
    plt.xlabel('Normalized Scale in $\sigma$',fontsize=20)
    plt.ylabel('Probability Density Function $P(r)$',fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(file_name+'_pos_log.png', transparent=True)
    plt.close()
    #for negative
    plt.figure(dpi=150)
    plt.plot(dist_x_neg,dist_y_neg,'.')
    plt.xlabel('Normalized Scale in $\sigma$',fontsize=20)
    plt.ylabel('Probability Density Function $P(r)$',fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(file_name+'_neg_log.png', transparent=True)
    plt.close()


def acf(acf_values,file_name,scale='log',):
    plt.figure(dpi=150)
    plt.plot(np.linspace(1,acf_values.size,acf_values.size),acf_values,'.')
    plt.ylim(1e-5,1.)
    if scale is 'linear':
        plt.ylim(-1.,1.)
    plt.xscale('log')
    plt.yscale(scale)
    plt.xlabel('lag $k$',fontsize=20)
    plt.ylabel('Auto-correlation',fontsize=20)
    plt.savefig(file_name+'.png',transparent=True)
    plt.close()
