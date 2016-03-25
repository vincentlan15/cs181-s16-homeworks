# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import random as random
from Perceptron import Perceptron
import time

# Implement this class
class KernelPerceptron(Perceptron):
	def __init__(self, numsamples):
		self.numsamples = numsamples
  		self.S=[]
  		self.alpha=[0]*99783

	# Implement this! 
#	def fit(self, X, Y):
#		self.X=X
#		self.Y = Y
#		itr=0
#		l=len(X)
#   
# 		while itr<self.numsamples:
#				t=randint(0,l-1)
#				yxt=0
#				for j in self.S:
#						yxt=yxt+self.alpha[j]*np.dot(X[j], X[t])
#      		if Y[t]*yxt<=0:
#						self.S=list(set().union(self.S,[t]))
#						self.alpha[t]=Y[t]
#      		itr=itr+1
# 		print(len(self.S))
# 		print(len(self.alpha))

	def fit(self, X, Y):
		self.X = X
		self.Y = Y
        # length = len(X)		
        
		itr = 0
        
		while itr < self.numsamples:
		    r = random.randint(0,len(X)-1)
		    yHat = 0
		    for i in self.S:
				  yHat += self.alpha[i] * np.dot(X[i], X[r])
                # print yHat
				  if Y[r] * yHat <= 0:
                #self.S.append(r)
		  		  		  self.S = list(set().union(self.S,[r]))
		  		  		  self.alpha[r] = Y[r]
		    itr+=1
            
# Implement this!
	def predict(self, X):
		XPrime=X
		l=len(XPrime)
		Y=[0]*l
		for t in (0,l-1):
		    for j in self.S:
				  Y[t]=Y[t]+self.alpha[j] * np.dot(self.X[j], XPrime[t])
		return (np.asarray(Y))

# Implement this class
class BudgetKernelPerceptron(Perceptron):
	def __init__(self, beta, N, numsamples):
		self.beta = beta
		self.N = N
		self.numsamples = numsamples
		
	# Implement this!
	# def fit(self, X, Y):

	# Implement this!
	# def predict(self, X):



# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]
X1 = X[:80000]
X2 = X[80000:]
Y1 = Y[:80000]
Y2 = Y[80000:]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 100
numsamples = 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
bef = time.time()
k = KernelPerceptron(numsamples)
k.fit(X1,Y1)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
#bk = BudgetKernelPerceptron(beta, N, numsamples)
#bk.fit(X, Y)
#bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=True)
