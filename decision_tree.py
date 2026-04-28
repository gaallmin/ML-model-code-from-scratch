import numpy as np
# y = [0, 0, 1, 1]
# probs = [0.5, 0.5]
# total_sum = 0.0
# gini_sum = 0.0
# for prob in probs:
#     total_sum += prob * np.log2(prob + 1e-12)
#     gini_sum += prob * (1-prob)
# print("entropy:", -total_sum)
# print("gini_sum:", gini_sum)

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))

def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return np.sum(probs * (1 - probs))

def impurity_gain(y_parent, y_left, y_right, criterion='entropy'):
    fn = entropy if criterion == 'entropy' else gini
    N  = len(y_parent)
    Nl, Nr = len(y_left), len(y_right)
    return fn(y_parent) - (Nl/N)*fn(y_left) - (Nr/N)*fn(y_right)

def best_split(X, y, criterion ='entropy'):
    best_gain, best_feature, best_threshold = -1, None, None
    N, V = X.shape
    for feature in range(V):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~ left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gain = impurity_gain(y, y[left_mask], y[right_mask], criterion)
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, feature, threshold
    return best_feature, best_threshold, best_gain

# ============================================

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
    
class DecisionTree:
    def __init__(self, max_depth=5, min_samples=2, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = criterion
    
    def fit(self, X, y):
        self.root = self._grow(X, y, depth=0)
        return self
    
    def _grow(self, X, y, depth):
        N,v = X.shape
        n_classes = len(np.unique(y))
        if (depth >= self.max_depth or N < self.min_samples or n_classes == 1):
            return TreeNode(prediction=self._majority_class(y))
        
        feature, threshold, gain = best_split(X, y, self.criterion)

        if feature is None or gain == 0:
            return TreeNode(prediction=self._majority_class(y))
        
        left_mask = X[:, feature] <= threshold
        print(f"Left mask : {left_mask}")

        left = self._grow(X[left_mask], y[left_mask], depth+1)
        right = self._grow(X[~left_mask], y[~left_mask], depth+1)
        print(f"Depth : {depth}, left : {left} - right: {right}")

        return TreeNode(feature=feature, threshold=threshold, left=left, right=right)
    
    def _majority_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]
    
    def _predict_one(self, x, node):
        if node.prediction is not None:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

# ============================================
X_toy = np.array([[1.0],
                   [2.0],
                   [3.0],
                   [7.0],
                   [8.0],
                   [9.0]])

y_toy = np.array([0, 0, 0, 1, 1, 1])

# fit
tree_toy = DecisionTree(max_depth=3, criterion='entropy')
tree_toy.fit(X_toy, y_toy)

# predict
print("Predictions:", tree_toy.predict(X_toy))
print("Actual:", y_toy)
print("Accuracy: ", tree_toy.accuracy(X_toy, y_toy))

# check what split was chosen
node = tree_toy.root
print(f"\nRoot split: feature={node.feature}, threshold={node.threshold}")
print(f"Left child prediction:  {node.left.prediction}")
print(f"Right child prediction: {node.right.prediction}")