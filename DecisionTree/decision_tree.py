import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, samples, attributes, target_attr):
        self.attributes = attributes
        self.target_attr = target_attr
        self.X, self.y = self._create_samples(samples)

    def fit(self):
        return self._grow_tree(self.X, self.y, [])

    def _grow_tree(self, X, y, block_attr):
        n_y = np.unique(y) #

        # check stopping criteria
        if len(n_y) == 1:
            return str(n_y[0])

        # find the best split
        best_idx, best_attribute = self._find_best_split(X, y, block_attr)
        if best_idx is None:
            return str(Counter(y).most_common(1)[0][0])

        block_attr.append(best_attribute)

        # grow tree
        tree = {str(best_attribute): {}}
        col_unique = np.unique(X[:, best_idx])

        for col in col_unique:
            n_samples = len(X)
            x_indices = [i for i in range(n_samples) if X[i, best_idx] == col]
            X_split = X[x_indices, :]
            y_split = y[x_indices]

            tree[best_attribute][str(col)] = self._grow_tree(X_split, y_split, block_attr.copy())

        return tree


    def _find_best_split(self, X, y, block_attr):
        best_gain = -1
        best_idx, best_attribute = None, None
        # for all attributes
        for idx, attr in enumerate(self.attributes):
            if attr in block_attr:
                continue
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
                best_gain = gain

        return best_idx, best_attribute

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    def _create_samples(self, samples):
        X = np.array([[sample[attr] for attr in self.attributes] for sample in samples])
        y = np.array([sample[self.target_attr] for sample in samples])
        return X, y


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    # Your code here
    decision_tree = DecisionTree(examples, attributes, target_attr)
    return decision_tree.fit()


if __name__ == "__main__":
    # samples = [
    #     {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'no'},
    #     {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'yes'},
    #     {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'no'},
    #     {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'yes'},
    #     {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak', 'PlayTennis': 'yes'},
    #     {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'PlayTennis': 'no'},
    #     {'Outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'PlayTennis': 'yes'},
    #     {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'no'},
    #     {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak', 'PlayTennis': 'yes'},
    #     {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Weak', 'PlayTennis': 'no'},
    #     {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Strong', 'PlayTennis': 'yes'},
    #     {'Outlook': 'Overcast', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'yes'},
    #     {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'Normal', 'Wind': 'Weak', 'PlayTennis': 'yes'},
    #     {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'yes'}
    # ]
    # samples = [
    #                 {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    #                 {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
    #                 {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    #                 {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'}
    #             ]
    samples =[ {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},
               {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
               {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
               {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},
               {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
               {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
               {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},
               {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'} ]
    
    attributes = ['Outlook', 'Wind']
    target_attr = 'PlayTennis'

    tree = DecisionTree(samples, attributes, target_attr)
    result = tree.fit()
    print(result)
    # print(tree.X, tree.y)
