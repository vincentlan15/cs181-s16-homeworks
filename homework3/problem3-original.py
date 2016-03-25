# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
from random import randint
from Perceptron import Perceptron
import time

# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples
        self.S = []
        self.alpha = [0]*99783

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        # length = len(X)		
        
        itr = 0
        
        while itr < self.numsamples:
            r = randint(0,len(X)-1)
            yHat = 0
            for i in self.S:
                yHat += self.alpha[i] * np.dot(X[i], X[r])
                # print yHat
            if Y[r] * yHat <= 0:
                #self.S.append(r)
                self.S = list(set().union(self.S,[r]))
                self.alpha[r] = Y[r]
            itr+=1
        print "Normal S length:",len(self.S)
            
    def predict(self, X):
        Xprime = X
        for idx, x in enumerate(Xprime):
            yHat = 0
            for j in self.S:
                yHat += self.alpha[j] * np.dot(self.X[j], x)
            Xprime[idx] = yHat
        return (Xprime[:, 0] >= 0)
        
    def test(self, X2, Y2):
        count = 0
        for idx, x in enumerate(X2):
            yHat = 0
            for j in self.S:
                yHat += self.alpha[j] * np.dot(self.X[j], x)
            print yHat>=0,Y[idx]
            if (yHat>=0):
                if Y[idx] == 1:
                    count+=1
            else:
                if Y[idx] == -1:
                    count+=1
        return float(count)/float(len(Y2))

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
            r = randint(0,len(X)-1)
            yHat = 0
            for i in self.S:
                yHat += self.alpha[i] * np.dot(X[i], X[r])
            if Y[r] * yHat <= self.beta:
                #self.S.append(r)
                self.S = list(set().union(self.S,[r]))
                self.alpha[r] = Y[r]
                
                if len(self.S) > N:
                    argmax = -999999
                    arg = 0
                    for j in self.S:
                        yHat2 = 0
                        for k in self.S:
                            yHat2 += self.alpha[k] * np.dot(X[j], X[k])
                        potential = Y[j] * (yHat2 - self.alpha[j] * np.dot(X[j], X[j]))
                        # print "POTENTIAL:",potential
                        if potential > argmax:
                            #print "POT:",potential
                            #print "NEW:",argmax
                            argmax = potential
                            arg = j
                    self.S.remove(arg)
                            
            itr+=1
        print "Budget S length:",len(self.S)
        
    def predict(self, X):
        Xprime = X
        for idx, x in enumerate(Xprime):
            yHat = 0
            for j in self.S:
                yHat += self.alpha[j] * np.dot(self.X[j], x)
            Xprime[idx] = yHat
        return (Xprime[:, 0] >= 0)
        
    def test(self, X2, Y2):
        count = 0
        for idx, x in enumerate(X2):
            yHat = 0
            for j in self.S:
                yHat += self.alpha[j] * np.dot(self.X[j], x)
            #Xprime[idx] = yHat
            #print yHat>=0,Y[idx]
            if (yHat>=0):
                if Y[idx] == 1:
                    count+=1
            else:
                if Y[idx] == -1:
                    count+=1
        return float(count)/float(len(Y2))


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
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
accuracy = k.test(X2,Y2)
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
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
accuracy = bk.test(X2,Y2)
aft = time.time()
print "Budget SVM time:",aft-bef
print "Budget Acc:",accuracy