import numpy as np


class SNL:

    def __init__(self, dIn: int, cOut: int):
        self.dIn = dIn
        self.cOut = cOut
        np.random.seed(42)
        self._W = np.random.randn(self.dIn)/np.sqrt(dIn-1)
        self._b = None


if __name__ == '__main__':
    snl = SNL(10, 10)
    print(snl._W)
