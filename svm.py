import numpy as np
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==== I. Soft margin, linear kernel, subgradient descent ====
class SVM:
    def __init__(self, lr, cost, epoch):
        self.lr = lr
        self.C = cost
        self.epoch = epoch
    
    def fit(self, X, y):
        # svm uses y in [-1,1]
        # assert : its condition holds from now on.
        assert set(y) == {-1,1}
        N, V = X.shape
        self.w = np.zeros(V)
        self.w0 = 0.0

        for _ in range(self.epoch):
            for i in range(N):
                margin = y[i] * (self.w0 + X[i] @ self.w)

                if margin >= 1:
                    # point outsie margin
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i])
                    self.w0 -= self.lr * (-self.C * y[i])
        return self
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def decision_function(self, X):
        return self.w0 + X @ self.w
    
    def predict(self, X):
        return np.sign(self.decision_function(X))
    # classify by sign of score <- {-1,1}

# ====== 2. Kernel SVM helper (RBF Kernel)  =============
def rbf_kernel(X1, X2, sigma=1.0):
    sq_dists = (np.sum(X1**2, axis=1, keepdims=True) # (N1,1)
                + np.sum(X2**2, axis=1) # (N2,)
                - 2 * X1 @ X2.T) # (N1,N2)
    return np.exp(-sigma * sq_dists)

def polynomial_kernel(X1, X2, degree=2, s=1.0, c=1.0):
    return (s * X1 @ X2.T + c) ** degree

# ========== PLOT DECISION BOUNDARY ==========
def plot_svm(model, X, y, title='SVM DECISION BOUNDARY'):
    fig, ax = plt.subplots(figsize=(7, 5))
    # 1. Build a grid covering a data
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:,0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:,0].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                           np.linspace(x2_min, x2_max, 200))
    grid = np.c_[xx1.ravel(), xx2.ravel()] # (200*200,2) ; grid coordinate
    z = model.decision_function(grid) #(40000,) # grid point flatten for the model
    z = z.reshape(xx1.shape) # (200, 200)

    # 2. Draw contour line
    ax.contourf(xx1, xx2, z, levels=[-999,0,999],
                alpha=0.15, colors=["#f28b82", "#82b4f2"])
    ax.contour(xx1, xx2, z,
            levels=[-1, 0, 1],
            colors=["blue", "black", "red"],
            linestyles=["--", "-", "--"],
            linewidths=[1.5, 2.0, 1.5])
    # 3: scatter the data points 
    ax.scatter(X[y == +1, 0], X[y == +1, 1],
               color="red",  edgecolors="k", s=60, label="Class +1")
    ax.scatter(X[y == -1, 0], X[y == -1, 1],
               color="blue", edgecolors="k", s=60, label="Class -1")

    # 4: highlight support vectors
    # support vectors = points where |decision_function| <= 1 + small tolerance
    scores = model.decision_function(X)
    sv_mask = np.abs(scores) <= 1.05       # (N,) boolean
    ax.scatter(X[sv_mask, 0], X[sv_mask, 1],
               s=200, facecolors="none", edgecolors="black",
               linewidths=2, label="Support vectors")

    ax.set_xlabel("X1"); ax.set_ylabel("X2")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()


# ========== LINEAR SVM ON SEPERABLE DATA ==========
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, random_state=42)
# converting label (0 -> -1)
y = np.where(y == 0, -1, 1)
'''
→ if condition is True, take x;
→ if condition is False, take y
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = SVM(lr=0.001, cost=1000, epoch=1000)
model.fit(X_train_s, y_train)

print(f"Train acc: {model.accuracy(X_train_s, y_train):.3f}")
print(f"Test  acc: {model.accuracy(X_test_s,  y_test):.3f}")
print(f"||w|| = {np.linalg.norm(model.w):.3f}  (margin width = {2/np.linalg.norm(model.w):.3f})")
# print(f"Decision function: {model.decision_function(X_train_s)}" )

plot_svm(model, X_train_s, y_train, title='Train')
plot_svm(model, X_test_s, y_test,title='Test')


# ========== KERNEL ON NON-LINEAR DATA ==========
X_nl, y_nl = make_circles(n_samples=200, noise=0.1, random_state=42)
y_nl = np.where(y_nl == 0, -1, 1)
K_rbf  = rbf_kernel(X_nl, X_nl, sigma=1.0)
K_poly = polynomial_kernel(X_nl, X_nl, degree=2)

print(f"\nRBF kernel  matrix shape:  {K_rbf.shape}")
print(f"Poly kernel matrix shape:  {K_poly.shape}")
print(f"K_rbf[0,0] = {K_rbf[0,0]:.3f} (self-similarity, always 1.0)")
print(f"K_rbf[0,1] = {K_rbf[0,1]:.3f} (similarity between sample 0 and 1)")
