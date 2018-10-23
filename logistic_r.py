import numpy as np
import matplotlib.pyplot as plt
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
        if(self.normalize):
            X = (X - self._min) / (self._max - self._min)
        if(self.add_bais):
            return np.where((1/(1+np.e**(-np.c_[np.ones(X.shape[0]), X].dot(th)))) > 0.5, 1, 0)
        else:
            return np.where(X.dot(self.th) > 0.5, 1, 0)





dt = np.loadtxt('data/data.csv',delimiter=',',skiprows=1)
np.random.shuffle(dt)
n = dt.shape[0]
x_train = dt[:int(n*.8),:2]
y_train = dt[:int(n*.8),2:]
x_test = dt[int(n*.8):,:2]
y_test = dt[int(n*.8):,2:]

lr = LogisticRegression(max_itr=100,toll=-1,learning_rate=1,normalize=False)
lr.fit(x_train,y_train)
th = lr.th

pv = lr.predict(x_test)
print('Original Value, Predicted Values for test data')
print(np.c_[y_test,pv])

acc = 1 - sum(abs(pv-y_test)) / (n-int(n*.8))
print('accuracy for test data : {}'.format(acc))
    

col= ['red' if l == 0 else 'green' for l in y_test]
plt.scatter(x_test[:,0],x_test[:,1],c=col)

y_vals = -(x_test[:,0] * th[1] + th[0])/th[2]
plt.plot(x_test[:,0], y_vals)

plt.show()
