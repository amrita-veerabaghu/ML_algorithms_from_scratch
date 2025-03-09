import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    # Your code here, make sure to round
    lin_reg = LinearRegressionGD(X, y, alpha, iterations)

    return lin_reg.fit()

class LinearRegressionGD:
    def __init__(self, X, y, alpha, iterations):
        """
        Linear regression using Gradient Descent
        :param X:
        :param y:
        :param alpha:
        :param iterations:
        """
        self.X = X
        self.y = y
        self.alpha = alpha
        self.iterations = iterations

        self.m, self.n = self.X.shape

        self.theta = np.zeros((self.n, ))
        print(self.X.shape, self.y.shape, self.theta.shape)

    def fit(self):

        for iter in range(self.iterations):
            self.theta = self.theta - self.alpha*(1/self.m)*self._gradient_theta(self.theta, self.X, self.y)

        return self.theta

    @staticmethod
    def _gradient_theta(theta, X, y):
        inside = np.dot(X, theta) - y # X m x n, theta n x 1
        gradient =  X.T.dot(inside)
        return gradient

if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([1, 2, 3])
    alpha = 0.01
    iterations=1000

    lin_reg = LinearRegressionGD(X, y, alpha, iterations)
    print(lin_reg.fit())