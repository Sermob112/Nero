import numpy as np
import matplotlib.pyplot as plt
def radial_basis_function(x, centers, widths):
    return np.exp(-(x - centers)**2 / (2 * widths**2))


# Обучающие данные
x_train = np.array([5, 6, 7, 8, 9])
y_train = np.sqrt(np.abs(x_train**2))
n_centers = 5
weights = np.random.rand(n_centers)
print(x_train)
print(y_train)
print(weights)
centers = np.linspace(1, 5, n_centers)
widths = 0.5
phi = radial_basis_function(x_train[:, np.newaxis], centers, widths)
print(phi)