# a) 50 Daten pro Art

import numpy as np
import matplotlib.pyplot as plt

# b)
iris = np.loadtxt(fname="iris.csv", delimiter=",")

# c)
X = iris[:, :-1]
T = iris[:, -1]
print(T)
# d)
X = X.T
print(X[:, 1])


# d)
plt.scatter(X[0], X[1])
plt.show()

# e)

plt.scatter(X[0], X[1], c=T, cmap=plt.cm.prism)
plt.show()
