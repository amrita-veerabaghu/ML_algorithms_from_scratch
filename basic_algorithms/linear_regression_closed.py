import numpy as np

class LinearRegression:
    def __init__(self, x: list[list[float]], y: list[float]):
        self.X = np.array(x)
        self.y = np.array(y)

        self.n_samples, self.n_features = self.X.shape

    def  _add_bias(self, x):
        ones = np.ones((self.n_samples, 1))
        return np.concat((ones,x), axis=1)

    def fit(self):
        # add bias column to data
        self.X = self._add_bias(self.X)

        # compute closed form solution
        xTx = np.dot(self.X.T, self.X)
        xTx_inv = np.linalg.inv(xTx)
        xTx_inv_xT = np.dot(xTx_inv, self.X.T)

        return np.dot(xTx_inv_xT, self.y)


if __name__ == "__main__":
    X = [[1, 1], [1, 2], [1, 3]] # doesn't work when we add bias support
    y = [1, 2, 3]

    lin_reg = LinearRegression(x = X, y = y)
    print(lin_reg.fit())