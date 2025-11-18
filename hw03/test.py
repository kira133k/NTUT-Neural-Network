import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def show_pattern(pattern_vector, title=""):
    image_matrix = pattern_vector.reshape(9, 5)  
    plt.imshow(image_matrix, cmap='binary')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])




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

pattern_names = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4", "Pattern 5", "Pattern 6", "Pattern 7", "Pattern 8", "Pattern 9", "Pattern A", "Pattern B", "Pattern C", "Pattern D", "Pattern E", "Pattern F"] 
total_simulations = len(patterns_for_homework)
simulation_count = 1



            
plt.figure(figsize=(12, 9))
plt.suptitle("All patterns diagram", fontsize=20)

for i in range(12):
    plt.subplot(3, 4, i+1)
    show_pattern(patterns_for_homework[i], f"Pattern {i+1}")

plt.tight_layout(rect=[0, 0.01, 1, 0.99])  
plt.savefig("all_patterns.png")
plt.close()



            
