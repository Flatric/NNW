import numpy as np

x = np.array([1., 2, 3])
W = np.array([[1., 2, 3], [4, 3, 2]])
print(W)

print(x.shape)
print(W.shape)

print(W.T)
print(x.T)  # seltsam ...? Logik: 1D bleibt 1D. x ist *kein* Spalten-Vektor!
y = x[np.newaxis].T  # erzeugt (a) 2D Matrix 1x3 (=Zeilenvektor!), (b) transponiert sie.
print(y)
print(y.shape)


class myclass:
    def __init__(self, W):  # Konstruktor, self entspricht this in Java/C++
        self._W = W  # Konvention: private Variablen beginnen mit _

    # self muss beim Zugriff angegeben werden, sonst
    # ist es eine lokale Variable
    def f(self):  # self muss immer explizit übergeben werden, sonst "static"
        return self._W * 2


m = myclass(np.array([-1, 1]))
print(m._W)  # Zugriff ist auch auf private Variablen erlaubt
m.f()  # m wird automatisch als "self" übergeben
