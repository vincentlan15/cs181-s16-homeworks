# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import random 
from Perceptron import Perceptron
import time

# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples
        self.S = []
        self.alpha = [0]*99783

    def fit(self, X, Y):
        self.X=X
        self.Y = Y
        itr=0
        l=len(X)
   
        while itr<self.numsamples:
            t=random.randint(0,l-1)
            yxt=0
            for j in self.S:
						yxt=yxt+self.alpha[j]*np.dot(X[j], X[t])
            if Y[t]*yxt<=0:
						self.S=list(set().union(self.S,[t]))
						self.alpha[t]=Y[t]
            itr=itr+1
#testing
#            print(len(self.S))
#            print(len(self.alpha))

    def predict(self, X):
        XPrime=X
        l=len(XPrime)
        Y=[0]*l
      #  print(l)
        for t in range(0,l-1):
       #     print(t)
            for j in self.S:
                Y[t]=Y[t]+self.alpha[j] * np.dot(self.X[j], XPrime[t])
            XPrime[t]=Y[t]
        return (XPrime[:,0]>=0)


        
    def test(self, X2, Y2):
        count = 0
        l=len(X2)
        for i in range(0,l-1):
            yHat = 0
            for j in self.S:
                yHat=yHat+self.alpha[j] * np.dot(self.X[j], X2[i])
            if (yHat>=0):
                if Y[i] == 1:
                    count=count+1
            else:
                if Y[i] == -1:
                    count=count+1
        return (float(count)/float(len(Y2)))

# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples
        self.S = []
        self.alpha = [0]*99783

    def fit(self, X, Y):
        self.X = X
        self.Y = Y		
        
        itr = 0
        
        while itr < self.numsamples:
            t = random.randint(0,len(X)-1)
            yxt = 0
            for i in self.S:
                yxt=yxt+self.alpha[i] * np.dot(X[i], X[t])
           # print(yxt)
            if Y[t] * yxt <= self.beta:
                self.S = list(set().union(self.S,[t]))
                self.alpha[t] = Y[t]
                if len(self.S) > N:
                    argmax = -999999
                    arg = 0
                    for j in self.S:
                        yxt2 = 0
                        for k in self.S:
                            yxt2=yxt2+self.alpha[k] * np.dot(X[j], X[k])
                        potential = Y[j] * (yxt2 - self.alpha[j] * np.dot(X[j], X[j]))
                        if potential > argmax:
                            argmax = potential
                            arg = j
                    self.S.remove(arg)
            itr=itr+1
        
    def predict(self, X):
        XPrime = X
        l=len(X)
        for i in range(0,l-1):
            yHat = 0
            for j in self.S:
                yHat=yHat+self.alpha[j] * np.dot(self.X[j], X[i])
            XPrime[i] = yHat
        return (XPrime[:, 0] >= 0)
        
        
    def test(self, X2, Y2):
        count = 0
        l=len(X2)
        for i in range(0,l-1):
            yHat = 0
            for j in self.S:
                yHat=yHat+self.alpha[j] * np.dot(self.X[j], X2[i])
            if (yHat>=0):
                if Y[i] == 1:
                    count=count+1
            else:
                if Y[i] == -1:
                    count=count+1
        return (float(count)/float(len(Y2)))



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
numsamples = 10000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
bef = time.time()
k = KernelPerceptron(numsamples)
k.fit(X1,Y1)
accuracy = k.test(X2,Y2)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
aft = time.time()
print "Normal SVM time:",aft-bef
print "Normal Acc:",accuracy

X = data[:, :2]
Y = data[:, 2]
X1 = X[:80000]
X2 = X[80000:]
Y1 = Y[:80000]
Y2 = Y[80000:]

bef = time.time()
bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X1, Y1)
accuracy = bk.test(X2,Y2)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
aft = time.time()
print "Budget SVM time:",aft-bef
print "Budget Acc:",accuracy