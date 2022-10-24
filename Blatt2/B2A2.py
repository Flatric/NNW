import numpy as np


class SNL:

    def __init__(self, dIn: int, cOut: int):
        self.dIn = dIn
        self.cOut = cOut
        np.random.seed(42)
        self._W = np.random.randn(self.dIn)/np.sqrt(dIn-1)
        self._b = np.zeros(self.cOut)[np.newaxis].T

    def neuron(self, X):
        net = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            for n in range(len(self._W)):
                net[i] += X[n, i] * self._W[n]
            net[i] += self._b[0]
        return net > 0

    def DeltaTrain(self, X, T, eta, maxIter, maxErrorRate):
        for i in range(maxIter):
            pass
        pass


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
    schwelle = 2
    net = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        for n in range(2):
            net[i] += X[n, i] * W[n]
    return net > schwelle

if __name__ == '__main__':
    snl = SNL(10, 1)
    snl._W = [-0.1, 1]
    iris = np.loadtxt(fname="/Users/jonathandeissler/Documents/NNW/Praktikum/Blatt1/iris.csv", delimiter=",")


    X = iris[:, :-1]
    T = iris[:, -1]
    X = X.T

    print(snl.neuron(X)==old_neuron(X))

