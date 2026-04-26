import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_hat):
    TN = np.sum((y_true == 0) & (y_hat == 0)) 
    TP = np.sum((y_true == 1) & (y_hat == 1))
    FN = np.sum((y_true == 1) & (y_hat == 0))
    FP = np.sum((y_true == 0) & (y_hat == 1))
    return TN, TP, FN, FP

def metrics_at_threshold(y_true, y_prob, threshold):
    # tau = threshold
    y_hat = (y_prob >= threshold).astype(int)
    TN, TP, FN, FP = confusion_matrix(y_true, y_hat)

    # accuracy :  how accurate it is : 
    accuracy = (TN + TP) / (TN + TP + FN + FP) 
    error = 1 - (FN + FP) / (TN + TP + FN + FP)

    # Recall : how much we can recall real positive among actual positive (y=1)
    recall = TP / (TP + FN) if (TP + FN) >0 else 0.0
    specificy = 1 - FP / (TN +FP) if (TN + FP) > 0 else 0.0

    # Precision : how precise among what model called as positive (y_hat = 1)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    fpr = FP / (TN + FP) if (TN + FP) > 0 else 0.0
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return dict(accuracy = accuracy, 
                error = error,
                recall = recall,
                specificy = specificy,
                precision = precision,
                f1 = f1,
                fpr = fpr,
                tpr = tpr)

def roc_curve(y_true, y_prob, n_thresholds = 100):
    thresholds = np.linspace(0, 1, n_thresholds)
    #print(np.linspace(0, 1,100)) # returns in an array
    fpr = []
    tpr = []
    for threshold in thresholds:
        metrics = metrics_at_threshold(y_true, y_prob, threshold)
        fpr.append(metrics['fpr'])
        tpr.append(metrics['tpr'])
    return np.array(fpr), np.array(tpr), thresholds

def auc(fpr, tpr):
    order = np.argsort(fpr) # sort by fpr so the curve goes left -> right
    auc = np.trapezoid(tpr[order], fpr[order])
    return auc

def pr_curve(y_true, y_prob, n_thresholds = 100):
    thresholds = np.linspace(0, 1, n_thresholds)
    precisions, recalls = [], []
    for threshold in thresholds:
        metrics = metrics_at_threshold(y_true, y_prob, threshold)
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
    return np.array(precisions), np.array(recalls), thresholds

'''
τ = 0.0  →  recall = 1.0  but precision = prevalence (near 0 for imbalanced)
            F1 will be low, sens+spec will be low

τ = 1.0  →  recall = 0.0, precision = undefined
            F1 = 0, sens+spec = 0 + 1 = 1? (misleading)
'''

def best_threshold_roc(y_true, y_prob, n_thresholds=100):
    # maximising sensitivity and specificity
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    scores = [metrics_at_threshold(y_true, y_prob, t)['recall'] +
              metrics_at_threshold(y_true, y_prob, t)['specificy']
              for t in thresholds]
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]

def best_threshold_f1(y_true, y_prob, n_thresholds=100):
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    scores = [metrics_at_threshold(y_true, y_prob, t)['precision'] +
              metrics_at_threshold(y_true, y_prob, t)['recall']
              for t in thresholds]
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]



# ============================ #
# TOY DATASET

X,y = make_classification(n_samples=500, n_features=5, weights=[0.5, 0.5], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# get prob from sklearn
clf = LogisticRegression().fit(X_train_s, y_train)
y_prob = clf.predict_proba(X_test_s)[:, 1] # p(Y=1)

# metrics at default
print("=== τ = 0.5 ===")
m = metrics_at_threshold(y_test, y_prob, threshold=0.5)
for k, v in m.items():
    print(f"  {k:12s}: {v:.3f}")

# ROC curve + AUC
fprs, tprs, thresholds = roc_curve(y_test, y_prob)
print(f"\nROC-AUC: {auc(fprs, tprs):.4f}")

# best thresholds
tau_f1,  f1_best  = best_threshold_f1(y_test, y_prob)
tau_roc, roc_best = best_threshold_roc(y_test, y_prob)
print(f"Best τ by F1:   {tau_f1:.3f}  (F1={f1_best:.3f})")
print(f"Best τ by sens+spec:  {tau_roc:.3f}  (sum={roc_best:.3f})")

# plot ROC
plt.figure(figsize=(6,5))
plt.plot(fprs, tprs, label=f'ROC (AUC={auc(fprs,tprs):.3f})')

plt.plot([0,1],[0,1],'--', label='Random')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title('ROC Curve')
plt.tight_layout(); plt.savefig('roc_curve.png', dpi=150); plt.show()