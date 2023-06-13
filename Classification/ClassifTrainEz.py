import numpy as np

def f(x):
    return 2/(1 + np.exp(-x)) - 1

def df(x):
    return 0.5*(1+x)*(1-x)

W1= np.array([[ 1.08857586 , 1.17121592 ,-2.06061722],
 [ 0.16099614 ,-0.18968963 ,-0.53662474]])
W2 = np.array([-3.30364048 ,-0.42555441])

def go_forward(inp):
    sum_hidden = np.dot(W1,inp)
    out = np.array([f(x) for x in sum_hidden])


    sum_hidden = np.dot(W2, out)
    y = f(sum_hidden)


    return  (y,out)

def Weight():
    return W1
def Weight2():
    return W2

def train(epoch):
    global W1,W2
    lmd = 0.01
    N = 10000
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0,count)]
        y,out = go_forward(x[0:3])
        e = y - x[-1]
        delta = e * df(y)
        W2[0] = W2[0] - lmd *delta *out[0]
        W2[1] = W2[1] - lmd * delta * out[1]

        delta2 = W2 * delta *df(out)

        W1[0, :] = W1[0, :] - np.array(x[0:3]) * delta2[0] * lmd
        W1[1, :] = W1[1, :] - np.array(x[0:3]) * delta2[1] * lmd

epoch = [
(-1,-1,-1,-1),
(-1,-1,1,1),
(-1,1,-1,-1),
(-1,1,1,1),
(1,-1,-1,-1),
(1,-1,1,1),
(1,1,-1,-1),
(1,1,1,-1)
]

train(epoch)
for x in epoch:
    y, out = go_forward(x[0:3])
    print(f"Выходное значение нС {y} => {x[-1]}")
    print(f"Веса для скрытого слоя: {Weight()}")
    print(f"Веса для выходного слоя: {Weight2()}")