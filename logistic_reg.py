# Logistic Regression
# Model probablity that binary outcome is 1 by squashing a linear function through sigmoid
# = Linear classifier in Log- odd Spaces
# Optimises the creoss entropy loss
# Linear Reg < Logistic Reg: when target is categorical, getting calibrated probability
# Logisitc Reg < SVM : when marginar boundary is needed


import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))
def cross_entropy(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)        # avoid log(0)
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

class LogisticRegression:
    def __init__(self, lr, epochs, lamb, penalty):
        self.lr = lr
        self.epochs = epochs
        self.lamb = lamb
        self.penalty = penalty
    
    def fit(self, x, y):
        N, V =  x.shape # N data, V features
        x_b = np.hstack([np.ones((N,1)), x])
        self.w = np.zeros(x_b.shape[1])
        self.loss_history = []

        for epoch in range(self.epochs):
            # 1. Linear combination
            z = x_b @ self.w
            # x_b shape : (240, 5)
            # w_shape : (5,)
            # z shape : (240, )

            # 2. Sigmoid (squashing between (0,1))
            p = sigmoid(z)

            # 3. Gradient
            # i) Error vector ; p-y
            error = p - y
            # error shape = (240, )
            # x shape = (240, 5)
            # ii) Grad Loss 1/N* x.T @e
            # ==> x_b.T @  error    =  grad_loss
            #      (5, 240) (240,)     (5,)
            grad_loss = 1/N * (x_b.T @ error)

            # iii) Penalty
            if self.penalty in ('L1', "Lasso"):
                # a) loss tracking
                weight_sum = np.sum(np.abs(self.w))
                new_loss = cross_entropy(y, p) + self.lamb * weight_sum
                self.loss_history.append(new_loss)

                # b) change gradient
                reg = self.lamb * np.sign(self.w)
                reg[0] = 0  # exclude bias from regularization
                grad_loss += reg
            
            elif self.penalty in ('L2', "Ridge"):
                # a) loss tracking
                weight_sum = np.sum(self.w**2)
                new_loss = cross_entropy(y, p) + self.lamb * weight_sum
                self.loss_history.append(new_loss)

                # b) change gradient
                reg = self.lamb * self.w
                reg[0] = 0
                grad_loss += reg
            else:
                self.loss_history.append(cross_entropy(y, p))

            #iv) weight update
            self.w -= self.lr * grad_loss
        return self
    
    def predict_prob(self, x):
        N = x.shape[0]
        x_b = np.hstack([np.ones((N, 1)), x])
        return sigmoid(x_b @ self.w)
    
    def predict(self, x, threshold = 0.5):
        return (self.predict_prob(x) >= threshold).astype(int)
    
    def accuracy(self, x, y):
        return np.mean(self.predict(x) == y)
        
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# reproducible 2-class data
X, y = make_classification(n_samples=300, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# always scale — logistic regression is sensitive to feature magnitude
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# baseline fit
model = LogisticRegression(lr=0.1, epochs=500, lamb=1.0, penalty='L2')
model.fit(X_train_s, y_train)

# X_train_s = scaler.transform(X_train)
print(f"Train acc: {model.accuracy(X_train_s, y_train):.3f}")
print(f"Testacc: {model.accuracy(X_test_s,  y_test):.3f}")
print(f"Final loss: {model.loss_history[-1]:.4f}")
print(f"Weights: {model.w.round(3)}")

'''
 == without scaling == 
big feature  → big gradient → big update → overshoots
small feature → tiny gradient → tiny update → barely moves

Train acc: 0.929
Testacc: 0.900
Final loss: 0.6250
Weights: [ 0.008 -0.263  0.204  0.273  0.005]
--> cannot compare -0.263 vs 0.204 — they're on different scales.

== with scaling ==
* covergence speed ->  stable,
* gradient steps -> even,
* weights comparable

Train acc: 0.929
Test  acc: 0.900
Final loss: 0.6351
Weights: [-0.019 -0.246  0.241  0.246 -0.   ]
'''
             
            
