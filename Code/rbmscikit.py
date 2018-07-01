# Inderster Florian, Universitaet Innsbruck 2018
# RBM for MNIST dataset
import matplotlib; matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import time
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

def loaddata(filename): # input string
	array = np.loadtxt("%s" %filename)
	return array

# main
nh = 100
lr = 0.01
nrows = 10
ncols = 10
epochs = 1000
ndata = 1000

print("loading data ...")
#X = loaddata('mnist01s')
idx = 0 #np.random.randint(1000-ndata-1)
X = loaddata('mnist01s')[idx:idx+ndata]
Y = np.arange(len(X))

print("training rbm ...	")
rbm = BernoulliRBM(n_components=nh,learning_rate=lr,n_iter=epochs,verbose=True)
rbm.fit(X)

# sampling
fig, axarr = plt.subplots(nrows,ncols,sharex = True)
plotevery = 10
for j in range(nrows):
	gibbs = np.array([X[j]])
	for i in range(ncols):
		gibbs = np.vstack((gibbs,rbm.gibbs(gibbs[i])))

		axarr[j,i].imshow(np.reshape(gibbs[i],(28,28)),cmap='Greys',\
									interpolation='none')
		axarr[j,i].axis('off')
		axarr[j,i].set_adjustable('box-forced')
		axarr[j,i].set_aspect(1)

		for u in range(plotevery):
			gibbs[i+1] = rbm.gibbs(gibbs[i+1])


plt.suptitle(r'$n_v = %i,\ n_h = %i,\ n_{data} = %i, lr = %.2f$' %(784,nh,ndata,lr),fontsize=20)
fig.subplots_adjust(0.08,0.02,0.92,0.85,0.08,0.23)
#plt.savefig('gibbs.pdf', bbox_inches='tight')
plt.show()

# plot hidden units
for i,comp in enumerate(rbm.components_):
	plt.subplot(10,10,i+1)
	plt.imshow(comp.reshape((28,28)), cmap='Greys',interpolation='none')
	plt.xticks(())
	plt.yticks(())
fig.subplots_adjust(0.08,0.02,0.92,0.85,0.08,0.23)
#plt.savefig('weights.pdf', bbox_inches='tight')
plt.show()

# plot log likelihood
#lik = rbm.score_samples(X)
#print(lik)
#plt.plot(np.arange(len(lik)),lik,'k-')
#plt.show()
