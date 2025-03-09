import numpy as np

"""
Code for Gradient Descent variations batch, SGD and mini-batch for Linear regression loss. 

General LR GD Formula: W = W - (2*lr* X^T.(W.X - y))/n_samples 
"""

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    # Your code here
    n_samples, n_features = X.shape

    if method == 'batch':
        for i in range(n_iterations):
            weights = weights - _update_weights(X, y, learning_rate, weights, n_samples)

    if method == 'stochastic':
        for i in range(n_iterations):
            for j in range(n_samples):
                x_j = X[j]
                y_j = y[j]
                weights = weights - _update_weights(x_j, y_j, learning_rate, weights, 1)



    if method == 'mini_batch':
        for i in range(n_iterations):
            for j in range(n_samples // batch_size):
                start, end = j * batch_size, (j + 1) * batch_size
                end = end if end <= n_samples else n_samples

                batch_X = X[start:end, :]
                batch_y = y[start:end]

                weights = weights - _update_weights(batch_X, batch_y, learning_rate, weights, batch_size)

    return weights

def _update_weights(x, y, lr, curr_weights, sample_size):

    return np.dot(x.T ,np.dot(x, curr_weights) - y) * 2 * lr / sample_size



if __name__=="__main__":
    X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    y = np.array([2, 3, 4, 5])
    weights = np.zeros(X.shape[1])
    learning_rate = 0.01
    n_iterations = 100
    output = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic')


    # output = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic')
    print(output)

