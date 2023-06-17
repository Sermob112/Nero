import random
import numpy as np
# np.exp(-1 / (2 * centers**2) * (centers - x) *(centers - x))
input_data = np.array([[0.2, 0.3], [0.5, 0.8], [0.1, 0.9], [0.3, 0.2], [0.7, 0.4], [0.6, 0.1], [0.9, 0.7], [0.8, 0.9]])
input_dim = (2,)
output_dim = (1,4)

# weights = np.random.rand(output_dim[0], output_dim[1], input_dim[0])
# distances = np.sum((weights - input_data[0]) ** 2, axis=(2,))
# print(distances.shape)
# print(np.argmin(distances))
# winner_index = np.unravel_index(np.argmin(distances),distances.shape)
# print(winner_index)
print(output_dim[0])
print(output_dim[1])