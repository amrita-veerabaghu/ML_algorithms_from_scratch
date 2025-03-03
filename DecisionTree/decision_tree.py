import numpy as np

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    # Your code here
    decision_tree = DecisionTree(examples, attributes, target_attr)
    return decision_tree.fit()

class DecisionTree:
    def __init__(self, samples, attributes, target_attr):
        self.attributes = attributes
        self.target_attr = target_attr
        self.X, self.y = self._create_samples(samples)

    def fit(self):
        return self._grow_tree(self.X, self.y)

    def _grow_tree(self, X, y):
        n_y = np.unique(y)

        # check stopping criteria
        if len(n_y) == 1:
            return n_y[0]

        # find the best split
        best_idx, best_attribute = self._find_best_split(X, y)

        # grow tree
        tree = {best_attribute: {}}
        col_unique = np.unique(X[:, best_idx])

        for col in col_unique:
            n_samples = len(X)
            x_indices = [i for i in range(n_samples) if X[i, best_idx] == col]
            X_split = X[x_indices, :]
            y_split = y[x_indices]

            tree[best_attribute][col] = self._grow_tree(X_split, y_split)

        return tree


    def _find_best_split(self, X, y):
        best_gain = -1
        best_idx, best_attribute = None, None
        # for all attributes
        for idx, attr in enumerate(self.attributes):
            col_unique = np.unique(X[:, idx])
            n_samples = len(X)
            parent_entropy = self._entropy(y)

            # find weighted child entropy
            child_entropies = []
            for col in col_unique:
                x_indices = [i for i in range(n_samples) if X[i, idx] == col]
                y_split = y[x_indices]
                weight = len(y_split)/len(y)
                child_entropies.append(weight*self._entropy(y_split))

            # find IG
            gain = parent_entropy - np.sum(child_entropies)

            # check for best_attr
            if gain > best_gain:
                best_attribute = attr
                best_idx = idx
        return best_idx, best_attribute

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    def _create_samples(self, samples):
        X = np.array([[sample[attr] for attr in self.attributes] for sample in samples])
        y = np.array([sample[self.target_attr] for sample in samples])
        return X, y
