import numpy as np
from decision_tree import TreeNode
from decision_tree import DecisionTree

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
# ====================================================

class RandomForest:
    def __init__(self, n_trees=100, max_depth=5, min_samples=2, max_features='sqrt', criterion='entropy'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.criterion = criterion
        self.trees = [] # storing all the fitted trees
        self.oob_score = None # out of bag accuracy
        '''
        using sqrt(v) is a rule of thumb, but it is importnat to decorrelate the trees, 
        to preserve the whole point of feature subsampling
        '''
    
    def _sample_features(self, V): # called once per tree
        if self.max_features == 'sqrt':
            m = max(1, int(np.sqrt(V)))
        elif self.max_features == 'log2':
            m = max(1, int(np.log2(V)))
        else:
            m = self.max_features # direct integer pass
        return np.random.choice(V, size=m, replace=False) 
    '''
    picks which features that tree is allowed to use at each split
    replace=False --> to have unique m distinct features
    '''

    def fit(self, X, y):
        N, V = X.shape
        self.trees = []

        oob_predictions =[[] for _ in range(N)]

        for b in range(self.n_trees):
            # === step 1. bootstrap sample ===
            boot_idx = np.random.choice(N, size=N, replace=True) #  it's sampling with replacement, meaning the same row can appear multiple times
            oob_idx = np.setdiff1d(np.arange(N), boot_idx) # arange(N) -> [0, 1,..., N]
            # find idx that are not in boot_idx 
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            # === step 2. building tree with feature subsampling ===
            feat_idx = self._sample_features(V)
            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples,
                                criterion=self.criterion)
            # overriding best_split to use subset of features
            tree.feature_indices = feat_idx
            original_grow = tree._grow
        
            def _grow_with_subset(X, y, depth, feat_idx=feat_idx):
                # only passing subset of features
                N_, V_ = X.shape
                n_classes = len(np.unique(y))
                if (depth >= tree.max_depth or N_ < tree.min_samples or n_classes == 1):
                    return TreeNode(prediction=tree._majority_class(y))
                
                # using m features
                feature, threshold, gain = best_split(X[:, feat_idx], y, tree.criterion)

                if feature is None or gain == 0:
                    return TreeNode(prediction=tree._majority_class(y))
                
                # mapping back to original feature index
                real_feature = feat_idx[feature]
                left_mask = X[:,real_feature] <= threshold

                left = _grow_with_subset(X[left_mask], y[left_mask], depth+1)
                right = _grow_with_subset(X[~left_mask], y[~left_mask], depth+1)

                return TreeNode(feature=real_feature, threshold=threshold, left=left, right=right)
            tree.roo = _grow_with_subset(X_boot, y_boot, depth=0)
            self.trees.append(tree)
            
            # step 3. out of bag predicitions
            if len(oob_idx) > 0:
                oob_preds = tree.predict(X[oob_idx])
                for i, pred in zip(oob_idx, oob_preds):
                    oob_predictions[i].append(pred)
            
            # =====================================
            # OOB score : majority vote across oob predictions
            oob_correct = 0
            oob_total = 0
            for i in range(N):
                if len(oob_predictions[i]) > 0:
                    majority = max(set(oob_predictions[i]), key=oob_predictions[i].count)
                    if majority == y[i]:
                        oob_correct += 1
                    oob_total += 1
            self.oob_score_ = oob_correct / oob_total if oob_total > 0 else None
            return self
        
    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([
            np.bincount(all_preds[:, i].astype(int)).argmax()
            for i in range(X.shape[0])
        ])
    

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=10,
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    test_size=0.2, random_state=42)

# single tree vs random forest
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)
print(f"Single tree test acc: {tree.accuracy(X_test, y_test):.3f}")

rf = RandomForest(n_trees=50, max_depth=5)
rf.fit(X_train, y_train)
print(f"Random forest test acc: {rf.accuracy(X_test, y_test):.3f}")
print(f"OOB score: {rf.oob_score_:.3f}")