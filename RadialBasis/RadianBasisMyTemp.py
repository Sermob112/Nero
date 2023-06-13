
import random
import numpy as np
import matplotlib.pyplot as plt

# Создаем пустой массив размером 5x6
W1 = [[0] * 6 for _ in range(5)]

# Заполняем массив случайными значениями от 0 до 1
for i in range(5):
    for j in range(6):
        W1[i][j] = random.random()
W1 = np.array(W1)
W2 = [random.random() for _ in range(5)]
W2 = np.array(W2)
# # Заполняем массив случайными значениями от 0 до 1
# for i in range(5):
#     W2[i][0] = random.random()

# Выводим полученный массив

# Определение радиально-базисной функции
def radial_basis_function(x, center, width):
    return np.exp(-(x - center)**2 / (2 * width**2))


# Обучающие данные
x_train = np.array([5, 6, 7, 8, 9, 10])
y_train = np.sqrt(np.abs(x_train**2))

# Обучение RBF-сети
def go_forward(inp):
    sum_hidden = np.dot(W1,inp)

    out = np.array([radial_basis_function(x, 5, 1) for x in sum_hidden])
    sum_hidden = np.dot(W2, out)
    y = radial_basis_function(sum_hidden,5,1)
    return  (y,out)




def train():
    global W1,W2
    lmd = 0.01
    N = 10000
    for k in range(N):
        y,out = go_forward(x_train)
        e = y - x_train
        gradient = -2 * np.dot(W1, e)
        W1 = W1.T - gradient * lmd
        W1 = W1.T
        gradient2 = -2 * np.dot(W1, e)
        W2 =- gradient2 * lmd

train()
x_test = np.linspace(4, 10, 6)  # Равномерно распределенные значения для построения графика
activations_test = np.zeros((len(x_test), 5))
for i, x in enumerate(x_test):
    for j in range(5):
        activations_test[i, j] = radial_basis_function(x, j, 1)

y,out = go_forward(x_train)
predictions = np.dot(activations_test, out)
plt.scatter(x_test, predictions, color='red', label='Аппроксимация RBF-сети')
plt.scatter(x_train, y_train, color='blue', label='Обучающие данные')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()