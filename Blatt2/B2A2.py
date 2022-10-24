import numpy as np
from Blatt1.nnwplot import plotTwoFeatures
from matplotlib import pyplot as plt

class SNL:

    def __init__(self, dIn: int, cOut: int):
        self.dIn = dIn
        self.cOut = cOut
        np.random.seed(42)
        self._W = np.random.randn(self.dIn) / np.sqrt(dIn - 1)
        self._b = np.zeros(self.cOut)[np.newaxis].T

    def neuron(self, X):
        net = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            for n in range(len(self._W)):
                net[i] += X[n, i] * self._W[n]
            net[i] += self._b[0]
        return net > 0

    def DeltaTrain(self, X, T, eta, maxIter, maxErrorRate):
        plotTwoFeatures(X, T, self.neuron)
        for i in range(maxIter):
            # delta_w = 1/n + sum eta (tn - yn) * xn
            iter_res = self.neuron(X)
            summe = np.sum(eta * (T - iter_res) * X, axis=1)
            delta_w = 1 / X.shape[1] * summe
            self._W += delta_w

            summe = np.sum(eta * (T - iter_res) * 1)
            delta_w = 1 / X.shape[1] * summe

            self._b += delta_w
            if i % 50 == 0:
                print(i)
                plt.ion()
                plotTwoFeatures(X, T, self.neuron)
                print(self._W)
                print(delta_w)
            if ErrorRate(iter_res, T) < maxErrorRate:
                plotTwoFeatures(X, T, self.neuron)
                print(ErrorRate(iter_res, T))
                print("yuhuu")
                return


def ErrorRate(Y, T):
    if Y.ndim == 1 or Y.shape[0] == 1:
        errors = Y != T
        return errors.sum() / Y.size
    else:  # für mehrere Ausgaben in one-hot Kodierung:
        # Dies brauchen Sie jetzt noch nicht nachzuvollziehen.
        errors = Y.argmax(0) != T.argmax(0)
        return errors.sum() / Y.shape[1]


def old_neuron(X):
    W = [-0.1, 1]
    schwelle = 0
    net = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        for n in range(2):
            net[i] += X[n, i] * W[n]
    return net > schwelle


if __name__ == '__main__':
    snl = SNL(2, 1)
#    iris = np.loadtxt(fname="/Users/jonathandeissler/Documents/NNW/Praktikum/Blatt1/iris.csv", delimiter=",")

#    X = iris[:, :-1]
#    T = iris[:, -1]
#    X = X.T

    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    X = X.T
    T = np.array([0, 0, 0, 1])
    print(X.shape)
    print(T.shape)
    print(X)
    print(T)

    snl.DeltaTrain(X, T, eta=0.01, maxIter=10000, maxErrorRate=0.000001)
