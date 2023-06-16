import random
import math
import matplotlib.pyplot as plt

N = [3]
Xn = 194
Yn = 114

L = len(N)
Ts = []
secondTs = []
classes = []
colors = ["row-cell__green", "row-cell__yellow", "row-cell__maroon", "row-cell__red", "row-cell__snow", "row-cell__purple", "row-cell__blue", "row-cell__lightGreen"]


def cArr(x, y):
    sumX = 0
    sumY = 0
    sumZ = 0
    step = 2

    for height in range(153, x, step):
        sumX += height ** 2

    for weight in range(45, y, step):
        sumY += weight ** 2

    for height in range(153, x, step):
        for weight in range(45, y, step):
            sumZ += weight / (height / 100) ** 2

    for height in range(153, x, step):
        for weight in range(45, y, step):
            ibm = weight / (height / 100) ** 2
            Ts.append([height / math.sqrt(sumX), weight / math.sqrt(sumY), ibm / math.sqrt(sumZ)])
            secondTs.append([height, weight, ibm])

    # print(sumX)
    # print(sumY)
    # print(sumZ)
    # print(Ts)

def rnd(min, max):
    return min + random.random() * (max - min)


class Neuron:
    def __init__(self, w, a):
        self.w = [rnd(0.093, 0.216) for _ in range(w)]


neurons = [[Neuron(3, 0) for _ in range(n)] for n in N]


def kohonen(a, w, y):
    for i in range(len(w)):
        w[i] += a * (y[i] - w[i])


def indexMinimum(D):
    index = 0
    min_val = D[index]
    for i in range(1, len(D)):
        if D[i] < min_val:
            index = i
            min_val = D[i]
    return index


def neuronWinner(y, layer=0):
    D = []
    for neuron in neurons[layer]:
        s = 0
        for i in range(len(y)):
            s += (y[i] - neuron.w[i]) ** 2
        D.append(math.sqrt(s))
    return indexMinimum(D)


def layerTraining(a, x):
    indexNeuron = neuronWinner(x)
    kohonen(a, neurons[0][indexNeuron].w, x)


def belong(x, index, action=1):
    global classes
    if action:
        classes = [] if not classes else classes
        indexNeuron = neuronWinner(x)
        classes[indexNeuron].append(secondTs[index][2])
    else:
        classes = [[] for _ in range(len(neurons[0]))]


def amountClasses():
    belong(0, 0, 0)
    for value in Ts:
        belong(value, Ts.index(value))
    return [len(value) for value in classes]


def learn(action=0, a=0.3, b=0.001, number=10):
    if action:
        while a > 0:
            for _ in range(1, number):
                for x in Ts:
                    layerTraining(a, random.choice(Ts))
            a -= b

    amountClasses()
    t = range(len(classes))
    classIndex = 0
    height = 153

    for row in t:
        weight = 45
        height += 2
        for cell in t:
            weight += 2
            ibm = weight / (height / 100) ** 2
            for values in classes:
                for value in values:
                    if value == ibm:
                        classIndex = classes.index(values)
            answer(classIndex, row, cell)

    drawChart()


def answer(index, x, y):
   print(index)
   print(x)
   print(y)


def drawChart():
    results = [['Iteration', 'Network Response']]
    indexTrain = 0
    for indexValue, value in enumerate(classes):
        for answer in value:
            results.append([indexValue, answer])
            indexTrain += 1

    plt.scatter([row[0] for row in results[1:]], [row[1] for row in results[1:]])
    plt.xlabel('Class')
    plt.ylabel('Body Mass Index')
    plt.title('Classification Results')
    plt.show()


cArr(Xn, Yn)
learn()