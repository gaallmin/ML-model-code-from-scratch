import numpy as np
from itertools import product

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logistic_reg import LogisticRegression

'''
HOLD OUT VALIDATION TUNING
split -> grid search -> best param
'''
def holdout_tune(model_class, param_grid, X,y, val_size=0.2, random_state=42):
    np.random.seed(random_state)
    N = len(y)

    # 1. spliting into train, validation
    idx = np.random.permutation(N)
    n_val = int(N * val_size)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # grid search over lambda
    # param_grid = {'C': [0.01, 0.1, 1, 10], 'max_iter': [100, 500]}

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_score, best_params = -1, None
    result = []

    for combo in product(*values): # * — unpack a list into separate arguments
        params = dict(zip(keys, combo))
        model = model_class(**params) # ** — unpack a dict into keyword arguments
        model.fit(X_tr, y_tr)
        score = model.accuracy(X_val, y_val)

        result.append((params, score))
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, result

'''
K fold cv: averaging over folds
'''
def kfold_tune(model_class, param_grid, X, y, k=5, random_state=42):
    np.random.seed(random_state)
    N = len(y)
    idx = np.random.permutation(N)

    # 1. splitting into K folds = list of k size arrays
    folds = np.array_split(idx, k)
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_score, best_params = -1, None
    results = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        fold_scores = []

        # for each fold
        # train on k-1, and validatioin on 1
        for fold_idx in range(k):
            val_idx = folds[fold_idx]
            tr_idx = np.concatenate(
                [folds[i] for i in range(k) if i != fold_idx]
            )

            # print(val_idx, tr_idx)
            # print("val shape",val_idx.shape)
            # print("tr shape",tr_idx.shape)

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = model_class(**params)
            model.fit(X_tr, y_tr)
            fold_scores.append(model.accuracy(X_val, y_val))

        # averaging across folds
        avg_score = np.mean(fold_scores)
        results.append((params, avg_score, np.std(fold_scores)))

        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    
    return best_params, best_score, results

# =============== RUNNING ========================

X, y = make_classification(n_samples=500, n_features=5, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# keep test set completely untouched — Fop slide 18
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# reuse your LogisticRegression from scratch
param_grid = {'lr': [0.001, 0.01, 0.1],
              'epochs': [200, 500],
              'lamb': [0.0, 0.1, 1.0],
              'penalty': ['ridge']}

# ── holdout tuning ─────────────────────────────────────────────────────
best_params_h, best_score_h, results_h = holdout_tune(
    LogisticRegression, param_grid, X_trainval, y_trainval, val_size=0.2)

print("=== Holdout Tuning ===")
print(f"Best params: {best_params_h}")
print(f"Best val score: {best_score_h:.3f}")

# ── k-fold tuning ──────────────────────────────────────────────────────
best_params_k, best_score_k, results_k = kfold_tune(
    LogisticRegression, param_grid, X_trainval, y_trainval, k=5)

print("\n=== 5-Fold CV Tuning ===")
print(f"Best params: {best_params_k}")
print(f"Best val score: {best_score_k:.3f}")

# ── final test — use best params, retrain on ALL trainval data ─────────
final_model = LogisticRegression(**best_params_k)
final_model.fit(X_trainval, y_trainval)
test_acc = final_model.accuracy(X_test, y_test)
print(f"\nFinal test accuracy: {test_acc:.3f}")