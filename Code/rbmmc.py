# Inderster Florian, Universitaet Innsbruck 2018
# RBM with Metropolis algorithm
import matplotlib; matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import random
import sys
# from pylab import *

def loaddata(filename): # input string
	array = np.loadtxt("%s" %filename)
	return array

def init(nv,nh,ndata):
	weights = np.random.rand(nv,nh)-.5
	hidden = np.random.randint(2,size=nh)*2-1 #np.ones(nh)-2
	shidden = np.random.randint(2,size=(ndata,nh))*2-1
	return [weights,hidden,shidden]

def trainmc(params,data,epochs,lr,T,err,plot):
	ndata = len(data)
	weights,hidden,shidden = params
	errorp = np.array([])
	errorerr = np.array([])
	plikep = []
	
	print('training started ...')
	for e in range(epochs):
		error = np.array([])
		plike = []
		# choose random index for visible config
		vrnd = np.random.random_integers(ndata)-1

		# update hidden layer
		for k in range(nh):
			# choose random hidden index
			hrnd = np.random.random_integers(nh)-1
			
			# compute energy for given config
			e1 = energy_h(params,data,hrnd)
			
			# flip chosen hidden unit
			shidden[vrnd,hrnd] *= -1 # 1-shidden[vrnd,hrnd]
			
			#if (0.< shidden[vrnd,hrnd] < 1.):
			if (0.<shidden[vrnd,hrnd]<1.):
				e2 = energy_h([weights,hidden,shidden],data,hrnd)
				de = e2-e1
				if de>0.:
					rnd = np.random.RandomState(1000).rand()
					if (np.exp(-de/T)>rnd):
						shidden[vrnd,hrnd] *= -1 # 1-shidden[vrnd,hrnd]
			else: shidden[vrnd,hrnd] *= -1

		# update weights
		for k in range(int(ndata/lr)):
			ri = np.random.random_integers(nv)-1
			rj = np.random.random_integers(nh)-1

			e1 = energy_wij(params,data,ri,rj)
			# increment weights
			dw = (np.random.rand()-0.5)
			weights[ri,rj] += dw

			if (-1.<weights[ri,rj]<1.):
				e2 = energy_wij([weights,hidden,shidden],data,ri,rj)
				de = e2-e1
				if (np.exp(-de/T)>np.random.rand() and de>0.):
					weights[ri,rj] -= dw
			else: weights[ri,rj] -= dw

		# use new weights
		params = [weights,hidden,shidden]

		# compute reconstruction error
		if err:
			for i in range(ndata):
				v1 = gibbs(params,data[i],1)
				error = np.append(error,1./4*np.sum((data[i]-v1)**2))
			print("Epoch %s: error is %.1f" % (e+1, np.mean(error)))
		else:
			sys.stdout.write("\rTraining Epoch: %i" %(e+1))
			sys.stdout.flush()
			#error = np.exp(1)

		errorp = np.append(errorp,np.mean(error))
		errorerr = np.append(errorerr,np.std(error)/np.sqrt(ndata))

		# pseudo log likelihood
		if err:
			for i in range(ndata):
				plike = np.append(plike,PL([weights,0,0],data[i]))
		plikep = np.append(plikep,np.mean(plike))

	# plot error to observe learning progress
	if plot:
		analysis(errorp,errorerr,plikep)

	return [weights,hidden,shidden] #[errorp,errorerr]
# use [errorp,errorerr] for MRE plots

def energy_h(params,data,x):
	weights,hidden,shidden = params
	e = 0
	ndata = len(data)
	for v in range(ndata):
		for i in range(nv):
			e += -weights[i,x]*data[v,i]*shidden[v,x]
			#e += -hidden[x]*shidden[v,x]
	return e

def energy_wij(params,data,x,y):
	weights,hidden,shidden = params
	e = 0
	ndata = len(data)
	for v in range(ndata):
		e += -weights[x,y]*shidden[v,y]*data[v,x]
	return e

def sample_v(params,data):
	w,h,sh = params
	hstates = np.zeros(nh)-1
	hprobs = sigm(np.dot(data,w))
	for i in range(nh):
		if hprobs[i] > np.random.RandomState(1000).rand(): hstates[i] = 1
	return hstates

def sample_h(params,data):
	w,h,sh = params
	vstates = np.zeros(nv)-1
	vprobs = sigm(np.dot(w,data.T))
	for i in range(nv):
		if vprobs[i] > np.random.RandomState(1000).rand(): vstates[i] = 1
	return vstates

def gibbs(params,data,steps):
	v0 = data
	for k in range(steps):
		h1 = sample_v(params,v0)
		v1 = sample_h(params,h1)
		v0 = v1
	return v0

def sigm(x):
	return 1./(1+np.exp(-x))

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

def analysis(errorp,recerr,plikep):
	# plot average reconstruction error each epoch
	#plt.subplot(121)
	fig,ax = plt.subplots()
	ax.set_yscale('log',nonposy='clip')
	#ax.set_xscale('log',nonposy='clip')
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

	plt.suptitle(r'$n_v = %i,\ n_h = %i,\ n_{data} = %i,\ lr = %.3f$' %(nv,nh,ndata,lr),fontsize=20)
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
			gibbsarr = np.vstack((gibbsarr,gibbs([params,0,0],gibbsarr[i],1)))
			axarr[i].imshow(np.reshape(gibbsarr[i],(28,28)),cmap='Greys',\
											interpolation='none')
			axarr[i].axis('off')
			axarr[i].set_adjustable('box-forced')
			axarr[i].set_aspect(1)

			for u in range(plotevery):
					gibbsarr[i+1] = gibbs([params,0,0],gibbsarr[i+1],1)

		#fig.subplots_adjust(0.08,0.02,0.92,0.85,0.08,0.23)
		#plt.savefig('gibbs.pdf', bbox_inches='tight')
	else:
		for j in range(nrows):
			gibbsarr = np.array([data[rndidx[j]]])
			for i in range(ncols):
				gibbsarr = np.vstack((gibbsarr,gibbs([params,0,0],gibbsarr[i],1)))
				axarr[j,i].imshow(np.reshape(gibbsarr[i],(28,28)),cmap='Greys',\
											interpolation='none')
				axarr[j,i].axis('off')
				axarr[j,i].set_adjustable('box-forced')
				axarr[j,i].set_aspect(1)

				for u in range(plotevery):
					gibbsarr[i+1] = gibbs([params,0,0],gibbsarr[i+1],1)

		fig.subplots_adjust(0.08,0.02,0.92,0.85,0.03,0.)
		#plt.savefig('gibbs.pdf', bbox_inches='tight')
	plt.show()

def plot_weights(weights):
	print('plotting weights ...')
	for i,comp in enumerate(np.transpose(weights[:,:100])):
		plt.subplot(10,10,i+1)
		plt.imshow(comp.reshape((28,28)), cmap='Greys',interpolation='none')
		plt.xticks(())
		plt.yticks(())
	#plt.savefig('weights.pdf', bbox_inches='tight')
	plt.show()

def plot_feature(weights):
	print('feature extraction ...')
	hid = np.zeros(nh)-1
	for i in range(nh):
		hid[i] *= -1
		von = sample_h([weights,0,0],hid)
		hid[i] *= -1

		plt.subplot(10,10,i+1)
		plt.imshow(von.reshape((28,28)), cmap='Greys',interpolation='none')
		plt.xticks(())
		plt.yticks(())
	#plt.savefig('weights.pdf', bbox_inches='tight')
	plt.show()


################### main ######################
nv,nh = 28*28,100
ndata = 2
epochs = 100
lr = 0.001

# mnist data set
print("loading data ...")
idx = 0
data = loaddata('mnist01s')[idx:idx+ndata]*2-1 #[0:10]*2-1


# mnist train with mc
ndata = len(data)
params = init(nv,nh,ndata)
params = trainmc(params,data,epochs,lr,1.,True,True)
weights,h,sh = params
plot_mnist(weights,data)
plot_weights(weights)
plot_feature(weights)

'''
# MRE after 200 epochs for various batch sizes depending on temperature
ndataarr = [1,4,5,10]
temp = [1./1000,1./500,1./100,1./50,1./25,1./10,1./5,1./2,1.,2.,10.,25.,50.,100.,500.,1000.,2000.,5000.]
colors = cm.YlGnBu(np.linspace(0.3,1,len(ndataarr)))
fig,ax = plt.subplots()
for u in range(len(ndataarr)):
	mre = np.zeros(len(temp))
	mrestd = np.zeros(len(temp))
	data = datar[0:ndataarr[u]]
	print('ndata = %i' %len(data))
	for i in range(len(temp)):
		print(temp[i])
		params = init(nv,nh,ndata)
		errv,std = trainmc(params,data,epochs,lr,temp[i],True,False)
		mre[i] = np.mean(errv[(epochs/2):])
		mrestd[i] = np.mean(std)*1./np.sqrt(len(errv[(epochs/2):]))
	ax.errorbar(temp,mre,yerr=mrestd,lw=1,marker='.',\
				ms='15',color=colors[u],label=r'$n_{data} = %i$' %ndataarr[u])


ax.set_xscale("log")
ax.set_yscale("log")
plt.xlabel(r'$T$',fontsize=20)
plt.xlim(1./2000,8000.)
plt.ylim(1.,1000.)
plt.ylabel(r'rec. error after %i epochs' %epochs,fontsize=16)
plt.suptitle(r'$n_v = %i,\ n_h = %i,\ lr = %.3f$' %(nv,nh,lr),fontsize=20)
plt.legend(loc='upper right',fontsize=14,numpoints=1)
plt.show()
'''

'''
# Plot of temperature against epochs
temp = [1./100,1./2,1.,2.,10.,100.]
mre = np.zeros(len(temp))
mrestd = np.zeros(len(temp))
fig,ax = plt.subplots()
colors = cm.YlGnBu(np.linspace(0.2,1,len(temp)))
for i in range(len(temp)):
	print(temp[i])
	params = init(nv,nh,ndata)
	errv,std = trainmc(params,data,epochs,lr,temp[i],True,False)
	plt.plot(np.arange(1,epochs+1),errv,color=colors[i],linewidth=2,label=r'$T=%.2f$' %temp[i])

#ax.set_xscale("log")
ax.set_yscale("log")
plt.ylabel(r'mean reconstruction error',fontsize = 16)
plt.xlabel(r'epoch',fontsize = 16)
plt.xlim(1,epochs)
plt.ylim(1./2,500.)
plt.legend(loc='upper right',fontsize=14,numpoints=1)
plt.suptitle(r'$n_v = %i,\ n_h = %i,\ n_{data} = %i,\ lr = %.3f$' %(nv,nh,ndata,lr),fontsize=20)
plt.show()
'''

'''
# Plot of the MRE depending on the amount of hidden units
# rec error vs nh
ndataarr = [1,4,5,10]
nhdata = [1,10,50,100,200,500,1000]
fig,ax = plt.subplots()
colors = cm.YlGnBu(np.linspace(0.3,1,len(ndataarr)))
for u in range(len(ndataarr)):
	errm = np.array([])
	stdm = np.array([])
	data = datar[0:ndataarr[u]]
	print('ndata = %i' %len(data))
	for i in range(len(nhdata)):
		nh = nhdata[i]
		print('nh = %i' %nh)
		params = init(nv,nh,ndataarr[u])
		errv,std = trainmc(params,data,epochs,0.001,1.,True,False)
		errm = np.append(errm,np.mean(errv[(epochs/2):]))
		stdm = np.append(stdm,np.mean(std[(epochs/2):])*1./np.sqrt(len(std[(epochs/2):])))
	ax.errorbar(nhdata,errm,yerr=stdm,lw=1.5,marker='.',ms='15',color=colors[u],\
				label=r'$n_{data} = %i$' %ndataarr[u])

ax.set_xscale("log")
ax.set_yscale("log")
plt.xlabel(r'$n_h$',fontsize=20)
plt.ylabel(r'rec. error after %i epochs' %epochs,fontsize=16)
plt.xlim(.6,1.5*max(nhdata))
plt.suptitle(r'$n_v = %i,\ epochs = %i,\ lr = %.3f$' %(nv,epochs,lr),fontsize=20)
plt.legend(loc='upper right',fontsize=14,numpoints=1)
plt.show()
'''
