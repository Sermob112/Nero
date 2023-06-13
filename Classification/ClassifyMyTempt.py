import numpy as np
import matplotlib.pyplot as plt



def f(x):
    return 2/(1 + np.exp(-x)) - 1

def df(x):
    return 0.5*(1+x)*(1-x)

W1= np.array([[ 0.3 , 0.5],
 [ 0.25 ,-0.18968963]])
W2 = np.array([-0.5 ,0.7])



def go_forward(inp):
    sum_hidden = np.dot(W1,inp)
    out = np.array([f(x) for x in sum_hidden])
    sum_hidden = np.dot(W2, out)
    y = f(sum_hidden)
    return  (y,out)

def train(epoch):
    global W1,W2
    lmd = 0.01
    N = 10000
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0,count)]
        y,out = go_forward(x[0:2])
        e = y - x[-1]
        delta = e * df(y)
        W2[0] = W2[0] - lmd *delta *out[0]
        W2[1] = W2[1] - lmd * delta * out[1]

        delta2 = W2 * delta *df(out)

        W1[0, :] = W1[0, :] - np.array(x[0:2]) * delta2[0] * lmd
        W1[1, :] = W1[1, :] - np.array(x[0:2]) * delta2[1] * lmd

def Weight():
    return W1
def Weight2():
    return W2

points = [
   (4,2,0),
    (8,8,1),
    (1,2,0),
    (6,6,1),
    (10,10,1),
    (1,1,0)
]
train(points)
for x in points:
    y, out = go_forward(x[0:2])
    print(f"Выходное значение нС {y} => {x[-1]}")
arr = []
count = len(points)
for i in range(count):
    x = points[i]
    y, out = go_forward(x[0:2])
    arr.append(x[0:2])
    if( y > x[-1]):
        plt.scatter(arr[i][0:1],arr[i][1:2],s=10, c='red')
    else:
        plt.scatter(arr[i][0:1], arr[i][1:2], s=10, c='blue')
slope = -W1[0, 0] / W1[0, 1]
intercept = -W2[0] / W1[0, 1]
x = np.linspace(0, 10)
y = slope * x + intercept
# plt.plot(x, y, color='red')
plt.grid(True)
plt.show()