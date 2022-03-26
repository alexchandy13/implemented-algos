import numpy as np
import math

class kNN:
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p
        self.data = []
        
    def fit(self, X, y):
        for i in range(len(X)):
            self.data.append([])
            for j in range(len(X[0])):
                self.data[i].append(X[i][j])
        for i in range(len(self.data)):
            self.data[i].append(y[i])
        
    def __classify(self, q):
        if len(q) != len(self.data[0]) - 1:
            return "Invalid Point"
        
        # Take L_p norms
        norms = []
        for i in range(len(self.data)):
            norm = 0
            for j in range(len(q)):
                norm += abs(q[j]-self.data[i][j]) ** self.p
            norm = norm ** (1/self.p)
            norms.append([norm,self.data[i][-1]])
        
        # Find k smallest norms
        norms = sorted(norms, key = lambda x: x[0])
        k_smallest = norms[:self.k]
        
        # Find most common class
        auth = 0
        for vote in k_smallest:
            if vote[1] == 0: auth += 1 
        
        # Return most common class        
        if auth > (self.k // 2):
            return 0
        else:
            return 1
        
    def predict(self, X, y):
        tp, fp, tn, fn = 0, 0, 0, 0    
        for i in range(len(X)):
            pred = self.__classify(X[i])
            if pred == 0 and y[i] == 0:
                tp += 1
            elif pred == 0 and y[i] == 1:
                fp += 1
            elif pred == 1 and y[i] == 1:
                tn += 1
            elif pred == 1 and y[i] == 0:
                fn += 1
        self.test_acc = (tp+tn)/(tp+fp+tn+fn)
        self.precision = tp/(tp+fp)
        self.recall = tp/(tp+fn)
        self.f1measure = (2*self.recall*self.precision)/(self.recall+self.precision)
        print('Test Accuracy: ' + str(self.test_acc))
        
        
class LogisticRegression:
    def __init__(self, lamb=0.1, eta=0.1, iters=100):
        self.lamb = lamb
        self.eta = eta
        self.iters = iters
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.w = np.zeros(X.shape[1] + 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        N = X.shape[0]
        
        for _ in range(self.iters):
            loss_sum = 0
            for i in range(N):
                xi = X[i]
                a = self.__sigmoid(np.matmul(xi,self.w))
                loss_sum += (a-y[i]) * xi
                
            grad = (1/N)*(loss_sum + self.lamb*self.w)
            self.w -= self.eta*grad
            
    def __classify(self, q):
        return np.round(self.__sigmoid(np.matmul(np.array(q),self.w[1:]) + self.w[0]))
    
    def __sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
    
    def predict(self, X, y):
        tp, fp, tn, fn = 0, 0, 0, 0    
        for i in range(len(X)):
            pred = self.__classify(X[i])
            if pred == 0 and y[i] == 0:
                tp += 1
            elif pred == 0 and y[i] == 1:
                fp += 1
            elif pred == 1 and y[i] == 1:
                tn += 1
            elif pred == 1 and y[i] == 0:
                fn += 1
        self.test_acc = (tp+tn)/(tp+fp+tn+fn)
        self.precision = tp/(tp+fp)
        self.recall = tp/(tp+fn)
        self.f1measure = (2*self.recall*self.precision)/(self.recall+self.precision)
        print('Test Accuracy: ' + str(self.test_acc))