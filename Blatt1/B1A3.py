import numpy as np
import nnwplot
import matplotlib.pyplot as plt

iris = np.loadtxt(fname="iris.csv", delimiter=",")
X = iris[:, :-1]
T = iris[:, -1]
X = X.T
X = X[:2, :]

print(X.shape)


def neuron(X):
    W = [-0.1, 1]
    schwelle = 2
    net = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        for n in range(2):
            net[i] += X[n, i] * W[n]
    return net > schwelle


nnwplot.plotTwoFeatures(X, T, neuron)
plt.figure()

# a) Bei einer Veränderung der Schwelle verschiebt sich die Trennlinie senkrecht zur Trennlinie

# b) Einer Veränderung der Gewichte verändert die Steigung der Trennlinie
