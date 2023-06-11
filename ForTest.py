import random
import numpy as np
# # Создаем пустой массив размером 5x6
# array = [[0] * 6 for _ in range(5)]
#
# # Заполняем массив случайными значениями от 0 до 1
# for i in range(5):
#     for j in range(6):
#         array[i][j] = random.random()
#
# # Выводим полученный массив
# for row in array:
#     print(row)
# W1 = [random.random() for _ in range(5)]
# print(W1)
# arr1 = np.array([5, 6, 7, 8, 9, 10])
#
# # Второй массив
# arr2 = np.array([[1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1]])
#
# # Умножаем и складываем значения
# result = np.sum(arr1 * arr2, axis=1)
# result = np.reshape(result, (-1, 1))
# print(result)
arr = np.array([[5, 6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15, 16],
                [17, 18, 19, 20, 21, 22],
                [23, 24, 25, 26, 27, 28],
                [29, 30, 31, 32, 33, 34]])

# Строка, которую нужно вычесть
subtract_row = np.array([1, 1, 1, 1, 1,1])

# Вычитаем строку из каждой строки массива
result = arr - subtract_row

# Выводим результат
print(result)