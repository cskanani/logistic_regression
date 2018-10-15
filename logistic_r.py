import numpy as np
class LogisticRegression:
    def __init__(self,normalize=True,add_bais=True,learning_rate=0.1,toll=0.0001,max_itr=100):
        self.normalize = normalize
        self.add_bais = add_bais
        self.learning_rate = learning_rate
        self.toll = toll
        self.max_itr = max_itr
        self.th = 0
        self._min = 0
        self._max = 0
        
    def fit(self,X,y):
        self._min = X.min()
        self._max = X.max()
        m = y.shape[0]
        
        if(self.normalize):
            X = (X - self._min) / (self._max - self._min)
        
        if(self.add_bais):
            X = np.c_[np.ones(m), X]
            
        th = np.zeros((X.shape[1],1))
        
        i = 0
        
        while (i < self.max_itr):
            th = th - (self.learning_rate / m) * (X.T).dot(1/(1+np.e**(-X.dot(th))) - y)
            i+=1
        
        self.th = th
        return th

    
    def predict(self,X):
        if(self.add_bais):
            X = (X - self._min) / (self._max - self._min)
            return 1/(1+np.e**(-np.c_[np.ones(X.shape[0]), X].dot(th)))
        else:
            X = (X - self._min) / (self._max - self._min)
            return X.dot(self.th)





dt = np.loadtxt('data/data.csv',delimiter=',',skiprows=1)
x_dt = dt[:,:2]
y_dt = dt[:,2:]

lr = LogisticRegression(max_itr=100,toll=-1,learning_rate=1,normalize=False)
lr.fit(x_dt,y_dt)
th = lr.th

print('Original Value, Predicted Values')
print(np.c_[y_dt,lr.predict(x_dt)])


import matplotlib.pyplot as plt

col= ['red' if l == 0 else 'green' for l in y_dt]
plt.scatter(x_dt[:,0],x_dt[:,1],c=col)

y_vals = -(x_dt[:,0] * th[1] + th[0])/th[2]
plt.plot(x_dt[:,0], y_vals)

plt.show()
