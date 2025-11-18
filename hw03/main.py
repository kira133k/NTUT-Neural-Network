import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import os

def show_pattern(pattern_vector, title=""):
    image_matrix = pattern_vector.reshape(9, 5)  
    plt.imshow(image_matrix, cmap='binary')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

def add_noise(pattern, noise_level=0.2):
    N = pattern.shape[0]
    num_to_flip = int(N * noise_level)
    
    indices_to_flip = np.random.choice(N, num_to_flip, replace=False)
    
    noisy_pattern = np.copy(pattern)
    
    for i in indices_to_flip:
        noisy_pattern[i] = -noisy_pattern[i]
        
    return noisy_pattern

def recall(noisy_pattern, W, max_iterations=5000):

    beta=100
    N = W.shape[0]
    s = np.copy(noisy_pattern)
    
    for _ in range(max_iterations):
        has_changed = False
        
        update_order = np.random.permutation(N)
        
        for i in update_order:
            u_i = np.dot(W[i, :], s)
            
            activation=np.tanh(beta * u_i/2.0)
            new_s_i = 1 if activation >= 0 else -1

            if s[i] != new_s_i:
                s[i] = new_s_i
                has_changed = True
                
        if not has_changed:
            break
            
    return s 

def similarity(p1, p2):
    return np.sum(p1 == p2) / len(p1)



# =========== Main function ===========
N = 45
W = np.zeros((N, N))

# --- 1.Read all pattern ---
pattern_directory = "picture/" 
file_paths = sorted(glob.glob(os.path.join(pattern_directory, "patten*.txt")))

pattern_list = []
for file_path in file_paths:
    pattern_vector = np.loadtxt(file_path, dtype=int)
    if pattern_vector.shape[0] == 45:
        pattern_list.append(pattern_vector)

patterns_for_homework = np.array(pattern_list[:12])

pattern_names = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4", "Pattern 5", 
                 "Pattern 6", "Pattern 7", "Pattern 8", "Pattern 9", "Pattern A", 
                 "Pattern B", "Pattern C", "Pattern D", "Pattern E", "Pattern F"] 
total_simulations = len(patterns_for_homework)

# --- 2.Weighting update method ---

# --- Hebbian Learning Method ---
# for p in patterns_for_homework:
#     p_vec = p.reshape(N, 1)
#     W += np.dot(p_vec, p_vec.T)

# np.fill_diagonal(W, 0)
# --- Hebbian Learning Method ---


# --- Pseudoinverse Matrix Method ---
X_pinv = np.linalg.pinv(patterns_for_homework.T)
W = patterns_for_homework.T @ X_pinv
np.fill_diagonal(W, 0)
# --- Pseudoinverse Matrix Method ---


noise_level = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
seed=[1, 2, 3, 4, 5 ,6 ,7 ,8 ,9 ,10]
sampled_seeds = random.sample(seed, 3)
for seeds in sampled_seeds:
    np.random.seed(seeds)
    random_idx = np.random.randint(0, len(patterns_for_homework))

    for noise in noise_level:
        print(f"\n--- Start simulation (Noise: {noise}, Seed: {seeds}, Total: {total_simulations} times) ---")        
        for idx in range(total_simulations):
            
            # output_dir = f"Hebbian_{str(noise)}_{str(total_simulations)}_SimulationResults"
            output_dir = f"Pseudoinverse_{str(noise)}_{str(total_simulations)}_SimulationResults"
            os.makedirs(output_dir, exist_ok=True) 

            p_clean = patterns_for_homework[idx]
            pattern_name = pattern_names[idx]

            p_noisy = add_noise(p_clean, noise)
            p_recovered = recall(p_noisy, W)
            
            plt.figure(figsize=(6, 3.5))
            plt.suptitle(f"Simulation: seed={seeds},noise={noise},{pattern_name},accuracy={similarity(p_clean, p_recovered):.2f}", fontsize=12)

            plt.subplot(1, 2, 1)
            show_pattern(p_noisy, "Received (Noisy)")
                        
            plt.subplot(1, 2, 2)
            show_pattern(p_recovered, "Recovered (Hopfield)")
                        
            plt.savefig(os.path.join(output_dir, f"sim_{seeds}_{noise}_{pattern_name}.png"))
            plt.close()
                        
            print(f"Simulation completed.")
            print(f"--- {total_simulations} times simulation already finished, result save in  '{output_dir}' folder ---")



            
