import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================
# SOM Definition
# ==========================================
class SOM:
    def __init__(self, map_size, input_dim, learning_rate=0.5, sigma=None, max_iter=100):
        self.map_height, self.map_width = map_size
        self.input_dim = input_dim
        self.lr = learning_rate
        self.max_iter = max_iter
        
        # Initial sigma default is the half size of map 
        self.sigma = sigma if sigma is not None else max(map_size) / 2.0
        
        # Initial weight randomly from data
        self.weights = np.random.rand(self.map_height, self.map_width, input_dim)


        # Build mesh map locations
        self.node_locations = np.zeros((self.map_height, self.map_width, 2))
        for i in range(self.map_height):
            for j in range(self.map_width):
                self.node_locations[i, j] = [i, j]
        
        self.errors = []

    def _get_sigma(self, t):
        return self.sigma * np.exp(-t / self.max_iter)

    def _find_winner(self, x):
        diff = self.weights - x
        dists_sq = np.sum(diff ** 2, axis=2)
        winner_idx = np.unravel_index(np.argmin(dists_sq), dists_sq.shape)
        return winner_idx

    def train(self, data):
        num_samples = data.shape[0]
        print(f"Start Training: Grid={self.map_height}x{self.map_width}, Epochs={self.max_iter}, Sigma={self.sigma}")

        for t in range(self.max_iter):
            indices = np.random.permutation(num_samples)
            total_error = 0
            
            sigma = self._get_sigma(t)

            for idx in indices:
                x = data[idx]
                
                win_i, win_j = self._find_winner(x)
                
                # Compute the error between sample and winner weight
                total_error += np.linalg.norm(x - self.weights[win_i, win_j])

                # Compute the distance between every neuron and the winner neuron
                winner_loc = self.node_locations[win_i, win_j]
                grid_dists_sq = np.sum((self.node_locations - winner_loc) ** 2, axis=2)
                
                # Alpha calculation
                alpha = np.exp(-grid_dists_sq / (2 * sigma ** 2))
                
                # Weighting update
                update_factor = self.lr * alpha[:, :, np.newaxis]
                self.weights += update_factor * (x - self.weights)
            
            self.errors.append(total_error / num_samples)

# ==========================================
# K-Means 
# ==========================================
def run_kmeans(filename, data, output_folder):
    output_path = f"{output_folder}/KMeans_{filename.split('.')[0]}.png"
    if os.path.exists(output_path):
        return

    k = 3 if "Three" in filename else 2
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.6, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=150, marker='x', linewidths=3, label='Centroids')
    plt.title(f"K-Means Comparison - {filename}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path)
    plt.close()
    print(f"K-Means result saved: {output_path}")

# ==========================================
# Experiment routine
# ==========================================
def ExperimentRoutine(filenames, map_size=(10,10), sigma=3.0):
    folder_path = "Results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    SOM_CONFIG = {
        'map_size': map_size,   
        'max_iter': 100,        
        'learning_rate': 0.05, 
        'sigma': sigma          
    }

    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found.")
            continue

        df = pd.read_csv(filename, sep=r'\s+', header=None)
        data = df.values

        # K-Means
        run_kmeans(filename, data, folder_path)

        som = SOM(input_dim=data.shape[1], **SOM_CONFIG)
        som.train(data)
        
        # Plotting results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Plot samples & neurons
        ax1.scatter(data[:, 0], data[:, 1], c='lightgreen', alpha=0.6, label='Samples')
        neurons = som.weights.reshape(-1, 2)
        ax1.scatter(neurons[:, 0], neurons[:, 1], c='red', s=30, label='Neurons')
        
        # Plot connections
        weights = som.weights
        for i in range(weights.shape[0]):
            ax1.plot(weights[i, :, 0], weights[i, :, 1], 'b-', alpha=0.6, linewidth=1)
        for j in range(weights.shape[1]):
            ax1.plot(weights[:, j, 0], weights[:, j, 1], 'b-', alpha=0.6, linewidth=1)
            
        ax1.set_title(f"SOM-{filename.split('.')[0]}")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. Convergence Curve
        ax2.plot(som.errors, color='purple', linewidth=2)
        ax2.set_title("Converge Curve")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Error Distance")
        ax2.grid(True)
        
        plt.tight_layout()
        
        output_file = f"{filename.split('.')[0]}_map={SOM_CONFIG['map_size']}_sigma={SOM_CONFIG['sigma']}_Result.png"
        plt.savefig(f"{folder_path}/{output_file}")
        print(f"result save at: {folder_path}/{output_file}")
        plt.close()

# ==========================================
# Main function
# ==========================================
if __name__ == "__main__":
    target_files = ['ThreeGroups.txt', 'TwoCircles.txt', 'TwoRings.txt']
    map_params = [(10, 10), (20,20), (30,30), (40,40), (50,50)]
    sigma_params = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    for map_size in map_params:
        for sigma in sigma_params:
            ExperimentRoutine(target_files, map_size=map_size, sigma=sigma)
    
    print("All Experiments Done!")