import numpy as np

W = np.array([[1,2,3],[3,4,5]])
print(W.shape)
print(W.shape[0])
print(W.shape[1])

print("W transposed", W.T)
print("W transposed shape",W.T.shape)

print(type(W))
print(type(W.shape))

r = np.arange(10)
print(r)
print(r.shape)

M=np.arange(12).reshape(3,4)
print(M)
print(M.shape)
print(M[2, 0])
print(M[1, 2])
print(M[1, :])
print(M[1])
print(M[:, 1])
print(M[:, [1]])
print(M[:, [3,0,1,1]])
print(M[:, 2:4])
print(M[-2,:])
print(M>4)
print(M[M>4])
M[M>4]=17
print(M)

s=0
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        s+=M[i,j]
        print(s,',',sep='',end='') # Parameter sep und end über Namen setzen
print(s)

def func(x):
    print(x*x)
    return x

print(func(3))
