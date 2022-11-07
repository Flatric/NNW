import numpy as np
from nnwplot import plotTwoFeatures
from matplotlib import pyplot as plt


class SNL:

    def __init__(self, dIn: int, cOut: int):
        self.dIn = dIn
        self.cOut = cOut
        np.random.seed(42)
        self._W = np.random.randn(cOut, dIn) / np.sqrt(dIn)
        self._b = np.zeros(cOut)[np.newaxis].T
        if cOut == 1:
            self.neuron = self.threshold
        else:
            self.neuron = self.thresholdMult

    def netsum(self, X):
        return self._W.dot(X) + self._b

    def threshold(self, X):
        return self.netsum(X) >= 0

    def DeltaTrain(self, X, T, eta, maxIter, maxErrorRate, show_iteration=10):
        plotTwoFeatures(X, T, self.neuron)
        plt.ion()
        for i in range(maxIter):

            iter_res = self.neuron(X)
            summe = (eta * (T - iter_res)) @ X.T
            delta_w = 1 / X.shape[1] * summe
            self._W += delta_w

            summe = np.sum(eta * (T - iter_res) * 1,axis=1)
            delta_w = 1 / X.shape[1] * summe
            delta_w = delta_w[np.newaxis].T
            self._b += delta_w

            if i % show_iteration == 0:
                print(f"Epoch {i}")
                print(f"Error rate {ErrorRate(iter_res, T)}")
                plotTwoFeatures(X, T, self.neuron)
                plt.show()

            if ErrorRate(iter_res, T) < maxErrorRate:
                plotTwoFeatures(X, T, self.neuron)
                print(self._W)
                plotTwoFeatures(X, T, self.neuron)
                print(f"Stop Early target Error rate reached: {ErrorRate(iter_res, T)}")
                plt.show()
                return

    def onehot(self, T):
        e = np.identity(self._W.shape[0])
        return e[:, T.astype(int)]

    def thresholdMult(self, X):
        to_onehot = self.netsum(X)
        res = np.argmax(to_onehot, axis=0)
        return self.onehot(res)


def ErrorRate(Y, T):
    if Y.ndim == 1 or Y.shape[0] == 1:
        errors = Y != T
        return errors.sum() / Y.size
    else:  # für mehrere Ausgaben in one-hot Kodierung:
        # Dies brauchen Sie jetzt noch nicht nachzuvollziehen.
        errors = Y.argmax(0) != T.argmax(0)
        return errors.sum() / Y.shape[1]


if __name__ == '__main__':
    snl = SNL(2, 3)

    iris = np.loadtxt(fname="iris.csv", delimiter=",")

    X = iris[:, :-1]
    T = iris[:, -1]
    T = snl.onehot(T)
    print(T.shape)
    X = X.T[2:, :]
    print(X.shape)

    snl.DeltaTrain(X, T, eta=0.005, maxIter=15000, maxErrorRate=0.05, show_iteration=350)
