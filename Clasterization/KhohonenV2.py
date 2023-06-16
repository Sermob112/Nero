import numpy as np
import matplotlib.pyplot as plt

class KohonenNetwork:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.rand(output_dim[0], output_dim[1], input_dim[0])

    def train(self, input_data, learning_rate, epochs):
        for epoch in range(epochs):
            for sample in input_data:
                winner_index = self.get_winner_neuron(sample)
                self.update_weights(sample, winner_index, learning_rate)

    def get_winner_neuron(self, sample):
        distances = np.sum((self.weights - sample) ** 2, axis=(2,))
        winner_index = np.unravel_index(np.argmin(distances), distances.shape)
        return winner_index

    def update_weights(self, sample, winner_index, learning_rate):
        neighborhood_radius = max(self.output_dim) / 2
        for i in range(self.output_dim[0]):
            for j in range(self.output_dim[1]):
                distance = np.sqrt((i - winner_index[0]) ** 2 + (j - winner_index[1]) ** 2)
                if distance <= neighborhood_radius:
                    influence = np.exp(-distance**2 / (2 * neighborhood_radius**2))
                    self.weights[i, j] += learning_rate * influence * (sample - self.weights[i, j])

    def get_clusters(self, input_data):
        clusters = []
        for sample in input_data:
            winner_index = self.get_winner_neuron(sample)
            clusters.append(winner_index)
        return clusters

    def visualize_clusters(self, input_data):
        clusters = self.get_clusters(input_data)
        plt.figure()
        for i, sample in enumerate(input_data):
            cluster_index = clusters[i]
            plt.scatter(sample[0], sample[1], color=f"C{cluster_index[0] * self.output_dim[1] + cluster_index[1]}")
        plt.title('Kohonen Clustering')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


# Example usage:
input_data = np.array([[0.2, 0.3], [0.5, 0.8], [0.1, 0.9], [0.3, 0.2], [0.7, 0.4], [0.6, 0.1], [0.9, 0.7], [0.8, 0.9]])
input_dim = (2,)
output_dim = (1, 3)

network = KohonenNetwork(input_dim, output_dim)
network.train(input_data, learning_rate=0.1, epochs=100)
network.visualize_clusters(input_data)