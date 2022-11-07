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
        plt.ion()
        for i in range(maxIter):

            iter_res = self.neuron(X)
            summe = np.sum(eta * (T - iter_res) * X, axis=1)
            delta_w = 1 / X.shape[1] * summe
            self._W += delta_w

            summe = np.sum(eta * (T - iter_res) * 1)
            delta_w = 1 / X.shape[1] * summe

            self._b += delta_w

            if i % 50 == 0:
                print(f"Epoch {i}")
                print(f"Error rate {ErrorRate(iter_res, T)}")
                plotTwoFeatures(X, T, self.neuron)
                plt.show()
            if ErrorRate(iter_res, T) < maxErrorRate:
                plotTwoFeatures(X, T, self.neuron)
                print(self._W)
                plotTwoFeatures(X, T, self.neuron)
                print(ErrorRate(iter_res, T))
                print("yuhuu")
                plt.show()
                return

    def onehot(self, T):
        e = np.identity(self._W.shape[0])
        print(e)
        return e[:, T.astype(int)]

def ErrorRate(Y, T):
    if Y.ndim == 1 or Y.shape[0] == 1:
        errors = Y != T
        return errors.sum() / Y.size
    else:  # für mehrere Ausgaben in one-hot Kodierung:
        # Dies brauchen Sie jetzt noch nicht nachzuvollziehen.
        errors = Y.argmax(0) != T.argmax(0)
        return errors.sum() / Y.shape[1]



if __name__ == '__main__':
    T = np.array([0, 2, 1, 2])
    snl = SNL(2, 1)
    print(snl.onehot(T))

