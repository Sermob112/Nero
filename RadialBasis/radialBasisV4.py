import numpy as np
import matplotlib.pyplot as plt
# Задайте входные данные
x_train = np.arange(1, 6, 1)  # Входные значения x
y_train = np.sqrt(np.abs(x_train ** 2))  # Фактические значения y

# Задайте параметры радиально-базисных функций
num_basis_functions = 5  # Количество радиально-базисных функций
centers = np.linspace(0.5, 5, num_basis_functions)  # Центры функций
width = 1.0  # Ширина функций (может быть задана константой)

# Инициализируйте веса нейросети случайными значениями
# weights = np.random.randn(num_basis_functions)


#Определите функцию радиально-базисной функции
def radial_basis_function(x, centers, width):
    return np.exp(-1 / (2 * width**2) * (centers - x) *(centers - x))

def redial_basis_func_dx(x, centers, width):
    return  -2 * (centers - x) * np.exp(-1 / (2 * width**2) * (centers - x) ** 2) / (2 * width**2)

#Функция активации
def relu(x):
    return np.maximum(0, x)
#Производная функции активации
def relu_derivative(x):
    return np.where(x > 0, 1, 0)
#
# # Вычислите выходы радиально-базисных функций для входных значений
weights = np.random.rand(num_basis_functions)
weights2 = np.random.rand(num_basis_functions)

# weighted_inputs = np.multiply(x_train, weights)
#
#
#
# print("Exit of uotputs")
# print(weighted_inputs)
# rbf_outputs = np.zeros((len(x_train), num_basis_functions))
# for i in range(len(x_train)):
#     for j in range(num_basis_functions):
#         rbf_outputs[i, j] = radial_basis_function(weighted_inputs[i], centers[j], width)

#Создаю 5 скрытых нейронов у каждого нейрона 5 значений, так как это радиально базисная нейросеть,
# то активациямми у меня будут выходы с радиально базисной функции
activations = np.zeros((len(x_train), num_basis_functions))
for i, x in enumerate(x_train):
    for j in range(num_basis_functions):
        activations[i, j] = radial_basis_function(x, j, 1)

#Создаю 5 выходных нейронов у каждого нейрона 5 весов, т.к. у нас 5 выходов со скрытых нейронов и заполняю их случайными значениями.
exit_neron_weights = np.zeros((len(x_train), num_basis_functions))
for i in range(len(x_train)):
    for j in range(num_basis_functions):
        exit_neron_weights[i, j] = np.random.rand()

# умножаю каждое входное значение на веса скрытого нейрона
# hidden_weighted_inputs = np.multiply(x_train, hidden_neron_weights)

# #Далее пропускаю каждый взвешанное входное значение через радиально базисную функцию
# hidden_neron_exit = np.zeros((len(x_train), num_basis_functions))
# for i in range(len(x_train)):
#     for j in range(num_basis_functions):
#         hidden_neron_exit[i, j] = radial_basis_function(hidden_weighted_inputs[i,j], centers[j],width)

#Получаю выход в виде суммы каждого скрытого нейрона
predicted_output_hidden_neurons = np.sum(activations, axis=1)

# #Далее умножаю выход каждого скрытого нейрона на веса выходного нейрона
# exit_weighted_inputs = np.multiply(predicted_output_hidden_neurons,exit_neron_weights)
#
#
# ######Добавить функцию активации!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# #Далее пропускаю каждый взвешанное входное значение через радиально базисную функцию
# exit_neron_exit = np.zeros((len(x_train), num_basis_functions))
# for i in range(len(x_train)):
#     for j in range(num_basis_functions):
#         exit_neron_exit[i, j] = relu(exit_weighted_inputs[i,j])
#
# #Получаю сумму в виде массива каждого выходного нейрона
# # predicted_output_exit_neurons = np.sum(exit_neron_exit, axis=1)






#Обучение весов с помощью градиентного спуска
learning_rate = 0.01  # Скорость обучения
num_epochs = 1000  # Количество эпох обучения

for epoch in range(num_epochs):

    # Далее умножаю выход каждого скрытого нейрона на веса выходного нейрона
    exit_weighted_inputs = np.multiply(predicted_output_hidden_neurons, exit_neron_weights)

    ######Добавить функцию активации!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Далее пропускаю каждый взвешанное входное значение через радиально базисную функцию
    exit_neron_exit = np.zeros((len(x_train), num_basis_functions))
    for i in range(len(x_train)):
        for j in range(num_basis_functions):
            exit_neron_exit[i, j] = relu(exit_weighted_inputs[i, j])

    # Получаю сумму в виде массива каждого  выходного нейрона
    predicted_output_exit_neurons = np.sum(exit_neron_exit, axis=1)

    # Вычислите ошибку и градиент
    #!!!!!!!!!!!!Здесь доделать!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    error = y_train - predicted_output_exit_neurons
    delta_output = error * relu_derivative(predicted_output_exit_neurons)
    gradient_output = np.dot(predicted_output_exit_neurons.T, delta_output)
    # gradient = -2 * np.dot(predicted_output_hidden_neurons, error)

    # Обновите веса
    # hidden_neron_weights += learning_rate * gradient_output
    exit_neron_weights += learning_rate * gradient_output

#
# Проверка аппроксимации на новых входных значениях
x_test = np.arange(1, 6, 1)  # Новые входные значения x
weighted_inputs_new = np.multiply(x_test, exit_neron_weights)

# Вычисление радиально-базисных функций на взвешенных новых входах
rbf_outputs_new = np.zeros((len(x_test), num_basis_functions))
print(weighted_inputs_new)
# def radial_basis_function_2(x, centers, width):
#     return np.exp(-1 / (2 * width**2) * (centers - x) *(centers - x))
# print(radial_basis_function_2(8.3,1,1))
# for i in range(len(x_test)):
#     for j in range(num_basis_functions):
#         rbf_outputs_new[i, j] = radial_basis_function_2(weighted_inputs_new[i], 3, width)

# Прогнозирование выходных значений на основе новых входов и весов
predicted_output_new = np.sum(rbf_outputs_new, axis=1)



plt.figure()
# plt.scatter(x_train, y_train, color='blue', label='Actual')
plt.plot(x_test, predicted_output_new, color='red', label='Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approximation of y = sqrt(abs(x^2))')
plt.legend()
plt.show()