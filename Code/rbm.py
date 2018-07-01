# Inderster Florian, Universitaet Innsbruck 2018
# RBM with contrastive divergence (k=1)
import matplotlib; matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import random
import sys
from sklearn.neural_network import BernoulliRBM

def loaddata(filename): # input string
	array = np.loadtxt("%s" %filename)
	return array

# initialize with random numbers
def init(nv,nh):
	np_rng = np.random.RandomState(1234)
	weights = np.asarray(np_rng.uniform(
			low=-0.1*np.sqrt(6./(nh+nv)),
            high=0.1*np.sqrt(6./(nh+nv)),
            size=(nv,nh)))
	vbias = np.zeros(nv)
	hbias = np.zeros(nh)
	return [weights,vbias,hbias]

def train(params,data,lr,err,plot):
	n = len(data)
	w, vbias, hbias = params
	k, vbiasj, hbiasj = params
	#rand = np.random.rand(n,n_h+1)
	
	errorp = np.array([])
	errorerr = np.array([])
	plikep = []

	for e in range(epochs):
		error = np.array([])
		plike = []
		dw,dv,dh = 0,0,0
		pos,neg = 0,0
		for i in range(n):
			v0 = data[i]
			h0prob = np.zeros(nh)
			v1prob = np.zeros(nv)
			h1prob = np.zeros(nh)
			h0state = np.zeros(nh)
			v1state = np.zeros(nv)
			
			# contrastive divergence CD1
			for j in range(nh):
				h0prob[j] = sigm(hbias[j] + np.dot(v0,w)[j])
				if h0prob[j] > np.random.rand(): h0state[j] = 1
			for j in range(nv):
				v1prob[j] = sigm(vbias[j] + np.dot(h0state,w.T)[j])
				if v1prob[j] > np.random.rand(): v1state[j] = 1
			for j in range(nh):
				h1prob[j] = sigm(hbias[j] + np.dot(v1prob,w)[j])

			# calculate delta wij und change of bias units
			posGrad = np.tensordot(v0,h0prob,axes=0)
			negGrad = np.tensordot(v1prob,h1prob,axes=0)
			dw = dw + (posGrad-negGrad)
			dv = dv + (v0-v1prob)
			dh = dh + (h0prob-h1prob)

			# get error to original data
			#v1 = gibbs([w,vbias,hbias],data[i],1)
			#error = np.append(error,np.sum((data[i]-v1)**2))
			#plike = np.append(plike,PL([w,vbias,hbias],data[i]))

		# adjust weights and bias units
		w = w + lr*dw/n
		vbias = vbias + lr*dv/n
		hbias = hbias + lr*dh/n

		params = [w, vbias,hbias]
		if err:
			for i in range(len(data)):
				v1 = gibbs(params,data[i],1)
				error = np.append(error,np.sum((data[i]-v1)**2))
			print("Epoch %s: error is %.1f" % (e+1, np.mean(error)))
		else:
			sys.stdout.write("\rTraining Epoch: %i" %(e+1))
			sys.stdout.flush()
			#error = np.exp(1)

		errorp = np.append(errorp,np.mean(error))
		errorerr = np.append(errorerr,np.std(error))


		# pseudo log likelihood
		if err:
			for i in range(ndata):
				plike = np.append(plike,PL(params,data[i]))
		plikep = np.append(plikep,np.mean(plike))

	if plot:
		analysis(errorp,errorerr,plikep,lr)

	return [w,vbias,hbias] #[errorp,errorerr]

# check which hidden units are on
def sample_v(params, data):
	w, vbias, hbias = params
	hstates = np.zeros(nh)
	hprobs = sigm(np.dot(data, w))
	for i in range(nh):
		if hprobs[i] > np.random.rand(): hstates[i] = 1
	return hstates

def sample_h(params, data):
	w, vbias, hbias = params
	vstates = np.zeros(nv)
	vprobs = sigm(np.dot(w,data.T))
	for i in range(nv):
		if vprobs[i] > np.random.rand(): vstates[i] = 1
	return vstates

def gibbs(params,data,steps):
	v0 = data
	for k in range(steps):
		h1 = sample_v(params,v0)
		v1 = sample_h(params,h1)
		v0 = v1
	return v0

def PL(params,x):
	np_rng2 = np.random.RandomState(1000)
	freex = free_energy(params,x)
	idx = np_rng2.randint(len(x))
	x[idx] = 1-x[idx]
	freexf = free_energy(params,x)
	cost = np.mean(nv*np.log(sigm(np.abs(freexf - freex))))
	return cost

def free_energy(params,x):
	w, vbias, hbias = params
	vb = np.dot(vbias,x)
	vh = hbias + np.dot(x,w)
	hb = np.sum(np.log(1+np.exp(vh)),axis=0)
	return -vb-hb

def sigm(x):
	return 1./(1+np.exp(-x))

def analysis(errorp,recerr,plikep,lr):
	# plot average reconstruction error each epoch
	#plt.subplot(121)
	fig,ax = plt.subplots()
	ax.set_yscale('log',nonposy='clip')
	plt.plot(np.arange(1,epochs+1),errorp,color='Blue',linewidth=2)
	plt.fill_between(np.arange(1,epochs+1),errorp-recerr,errorp+recerr,\
					alpha=0.5,facecolor='LightBlue',edgecolor='LightBlue',antialiased=True)
	plt.ylabel(r'mean reconstruction error',fontsize = 16)
	plt.xlabel(r'epoch',fontsize = 16)
	plt.xlim(1,epochs)
	plt.grid(True)

	'''
	plt.subplot(122)
	plt.plot(np.arange(1,epochs+1),plikep,color='Grey',linewidth=3)
	plt.ylabel(r'pseudo log likelihood',fontsize = 16)
	plt.xlabel(r'epoch',fontsize = 16)
	plt.xlim(1,epochs)
	plt.grid(True)
	plt.tight_layout(pad=3)
	'''

	plt.suptitle(r'$n_v = %i,\ n_h = %i,\ n_{data} = %i,\ lr = %.2f$' %(nv,nh,ndata,lr),fontsize=20)
	plt.show()

def plot_mnist(params,data):
	nrows = np.amin([ndata,5])
	ncols = 10
	print('plotting ...')

	plotevery = 1
	rndidx = random.sample(range(ndata),ndata)
	fig, axarr = plt.subplots(nrows,ncols,sharex=True)
	if len(data) == 1:
		gibbsarr = np.array([data[rndidx[0]]])
		for i in range(ncols):
			gibbsarr = np.vstack((gibbsarr,gibbs(params,gibbsarr[i],1)))
			axarr[i].imshow(np.reshape(gibbsarr[i],(28,28)),cmap='Greys',\
											interpolation='none')
			axarr[i].axis('off')
			axarr[i].set_adjustable('box-forced')
			axarr[i].set_aspect(1)

			for u in range(plotevery):
				gibbsarr[i+1] = gibbs(params,gibbsarr[i+1],1)

		#fig.subplots_adjust(0.08,0.02,0.92,0.85,0.08,0.23)
		#plt.savefig('gibbs.pdf', bbox_inches='tight')
	else:
		for j in range(nrows):
			gibbsarr = np.array([data[rndidx[j]]])
			for i in range(ncols):
				gibbsarr = np.vstack((gibbsarr,gibbs(params,gibbsarr[i],1)))
				axarr[j,i].imshow(np.reshape(gibbsarr[i],(28,28)),cmap='Greys',\
											interpolation='none')
				axarr[j,i].axis('off')
				axarr[j,i].set_adjustable('box-forced')
				axarr[j,i].set_aspect(1)

				for u in range(plotevery):
					gibbsarr[i+1] = gibbs(params,gibbsarr[i+1],1)

		fig.subplots_adjust(0.08,0.02,0.92,0.85,0.03,0.)
		#plt.savefig('gibbs.pdf', bbox_inches='tight')
	plt.show()

def plot_weights(params):
	print('plotting weights ...')
	for i,comp in enumerate(np.transpose(params[0])):
		plt.subplot(10,10,i+1)
		plt.imshow(comp.reshape((28,28)), cmap='Greys',interpolation='none')
		plt.xticks(())
		plt.yticks(())
	#plt.savefig('weights.pdf', bbox_inches='tight')
	plt.show()

def plot_feature(params):
	print('feature extraction ...')
	hid = np.zeros(nh)
	for i in range(nh):
		hid[i] += 1
		von = sample_h(params,hid)
		hid[i] -= 1

		plt.subplot(10,10,i+1)
		plt.imshow(von.reshape((28,28)), cmap='Greys',interpolation='none')
		plt.xticks(())
		plt.yticks(())
	#plt.savefig('weights.pdf', bbox_inches='tight')
	plt.show()


################### main ######################
nv,nh = 28*28,100
ndata = 5
epochs = 50
lr = 0.1

# mnist data set
print("loading data ...")
idx = 5 #np.random.randint(1000-ndata-1)
data = loaddata('mnist01s')[idx:idx+ndata]

# mnist
params = init(nv,nh)
params = train(params,data,lr,True,True)
plot_mnist(params,data)
plot_weights(params)
plot_feature(params)

'''
# rec error vs nh
# for plotting the dependency of the MRE on the size of hidden layer
nhdata = [1,10,50,100,200,500,1000]
errm = np.array([])
stdm = np.array([])
for i in range(len(nhdata)):
	nh = nhdata[i]
	print(nh)
	params = init(nv,nh)
	errv,std = train(params,data,0.1,True,False)
	errm = np.append(errm,np.mean(errv[(9*epochs/10):]))
	stdm = np.append(stdm,np.mean(std)*1./np.sqrt(len(std)))

fig,ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.errorbar(nhdata,errm,yerr=stdm,fmt='.',ms='15',color='Black')
plt.xlabel(r'$n_h$',fontsize=20)
plt.ylabel(r'rec. error after %i epochs' %epochs,fontsize=16)
plt.xlim(.6,1.5*max(nhdata))
plt.suptitle(r'$n_v = %i,\ epochs = %i,\ n_{data} = %i,\ lr = %s$' %(nv,epochs,ndata,lr),fontsize=20)
plt.show()
'''
