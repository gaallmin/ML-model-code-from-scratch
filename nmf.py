# NMF approximates a data matrix X as w@H
# W contians learned approximates
# H tells how much each approximates contributes to each data point

# it optimizes reconstruction error ; How clos W@H is to X
# error  =  Frobenius or KL-divergence loss

# NVF > PCA : when the data is non-negative (e.g. images, text, counts)
# K-means > NVF : hard cluster > soft weightings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces


class NMF:
    def __init__(self, K=3, n_iter=200, tolerance=1e-4, random_state=42):
        self.K = K # number of basis vectors (latent components)
        self.n_iter = n_iter
        self.tol = tolerance
        self.random_state = random_state
    
    def fit(self, X):
        # X shape (v,n) : features = rows and samples= columns
        np.random.seed(self.random_state)
        V, N = X.shape

        # initialising W and H with non-negative value
        self.W = np.abs(np.random.randn(V, self.K)) * 0.1
        self.H = np.abs(np.random.randn(self.K, N)) * 0.1

        # w : (V, K) : basis matrix, each column = one archetype
        # h : (K, N) : coefficient matrix, each  column = one data point

        self.loss_history = []
        eps = 1e-10 # just to prevent division by zero

        for t in range(self.n_iter):
            # 1. UPDATING H
            numerator_H = self.W.T @ X # (K,N)
            denominator_H = self.W.T @ self.W @ self.H + eps # (K,N)
            self.H *= numerator_H / denominator_H # (K,N)

            # 2. UPDATING W
            numerator_W = X @ self.H.T # (V, K)
            denominator_W = self.W @ self.H @ self.H.T + eps # (V, K)
            self.W *= numerator_W / denominator_W # (V, K)

            # tracking reconstruction loss
            loss = np.linalg.norm(X - self.W @ self.H, 'fro') ** 2 # Frobenius norm
            self.loss_history.append(loss)

            # convergence check
            if t > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                print(f"Converged at iteration {t}")
                break
            
        return self

    def reconstruct(self, X=None): # approximation of original data
        return self.W @ self.H # (V,N)
    
    def encode(self, X): # project new data onto learned basis W
        return np.linalg.pinv(self.W) @ X #(K,N)
        '''
        cf) np.linalg.pinv : The pseudo-inverse of a matrix A, denoted 
        , is defined as: “the matrix that ‘solves’ 
        [the least-squares problem] Ax=b
        '''   
    def frobenius_loss(self, X):
        return np.linalg.norm(X - self.W @ self.H, 'fro') ** 2
 
# =========================
np.random.seed(42)

V, N, K_true = 50, 200, 5
W_true = np.abs(np.random.randn(V, K_true))
H_true = np.abs(np.random.randn(K_true, N))
X = W_true @ H_true + 3.0 * np.abs(np.random.randn(V, N))

# Trying negative value
X_neg = X - 5.0   # shift to include negatives
model_neg = NMF(K=3)
model_neg.fit(X_neg)
print(model_neg.loss_history[-1])  # does it converge?

# # === ONE K == 
# model = NMF(K=10, n_iter=500, tolerance=1e-10)
# model.fit(X)
# print(f"Iterations run: {len(model.loss_history)}")
# print(f"First loss: {model.loss_history[0]:.2f}")
# print(f"Last loss:  {model.loss_history[-1]:.2f}")

# print(f"\nW shape: {model.W.shape}   (V x K basis matrix)")
# print(f"H shape: {model.H.shape}   (K x N coefficient matrix)")
# print(f"Final loss: {model.loss_history[-1]:.4f}")

# # reconstruction quality
# X_hat = model.reconstruct()
# rel_error = np.linalg.norm(X - X_hat, 'fro') / np.linalg.norm(X, 'fro')
# print(f"Relative reconstruction error: {rel_error:.4f}")
# V, N, K = X.shape[0], X.shape[1], 3
# print(f"Original storage:    {V*N}")
# print(f"Compressed storage:  {V*K + K*N}")
# print(f"Compression ratio:   {V*N / (V*K + K*N):.2f}x")
# print(f"Compressed? {K < (V*N)/(V+N)}")

# # plot loss curve
# plt.plot(model.loss_history)
# plt.xlabel('Iteration')
# plt.ylabel('Frobenius loss')
# plt.title('NMF Convergence')
# plt.tight_layout()
# plt.show()

# # ── compare different K values ─────────────────────────────────────────────
# K_values = [1, 2, 3, 5, 8, 10, 15, 20]
# results = []

# for K in K_values:
#     model = NMF(K=K, n_iter=500, tolerance=1e-10)
#     model.fit(X)
#     X_hat = model.reconstruct()
#     rel_err = np.linalg.norm(X - X_hat, 'fro') / np.linalg.norm(X, 'fro')
#     n_iter  = len(model.loss_history)
#     results.append((K, model.loss_history[-1], rel_err, n_iter))
#     print(f"K={K:2d}  loss={model.loss_history[-1]:8.2f}  "
#           f"rel_error={rel_err:.4f}  iterations={n_iter}")

# # ── plot relative error vs K ───────────────────────────────────────────────
# Ks    = [r[0] for r in results]
# errs  = [r[2] for r in results]
# iters = [r[3] for r in results]

# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# # plot 1 — reconstruction error vs K
# axes[0].plot(Ks, errs, 'bo-', linewidth=2, markersize=8)
# axes[0].axvline(x=K_true, color='red', linestyle='--',
#                 label=f'True K={K_true}')
# axes[0].set_xlabel('K (number of components)')
# axes[0].set_ylabel('Relative reconstruction error')
# axes[0].set_title('Reconstruction Error vs K')
# axes[0].legend()

# # plot 2 — iterations to converge vs K
# axes[1].bar(Ks, iters, color='steelblue')
# axes[1].set_xlabel('K (number of components)')
# axes[1].set_ylabel('Iterations to converge')
# axes[1].set_title('Convergence Speed vs K')

# plt.tight_layout()
# plt.show()