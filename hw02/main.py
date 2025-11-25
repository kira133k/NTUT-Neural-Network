import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy # 用於儲存最佳模型

# --- 1. CONFIGURATION ---

# Initial Setting
LEARNING_RATE = 0.01
MAX_ITERATION = 10000
BATCH_SIZE = 32
SEED = 42

# Early Stopping Setting
STOP = 100
BEST_VAL_LOSS = float('inf')
STOP_COUNTER = 0

# # Model Setting
# # # --- Iris --- 
# DATA_FILE = 'HW2_MLP_dataset/iris.txt'
# INPUT_FEATURES = 4
# OUTPUT_CLASSES = 3
# HIDDEN_LAYERS_CONFIG = [5]
# DROP = 0.4 #0.3 0.4 0.4
# DECAY = 1e-4 #4 4 4

# # --- Breast ---
# DATA_FILE = 'HW2_MLP_dataset/breast-cancer-wisconsin.txt'
# INPUT_FEATURES = 9
# OUTPUT_CLASSES = 2
# HIDDEN_LAYERS_CONFIG = [8]
# DROP = 0.4 #0.3 0.4 0.4
# DECAY = 1e-2 #4 2 2


# --- Wine ---
DATA_FILE = 'HW2_MLP_dataset/wine.txt'
INPUT_FEATURES = 13
OUTPUT_CLASSES = 3
HIDDEN_LAYERS_CONFIG = [11]
DROP = 0.25
DECAY = 1e-4


# CUDA Setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. UTILITIES ---

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    
# --- 3. MODEL DEFINITION ---

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_p=DROP):
        super(MLP, self).__init__()
        
        layers = []
        current_input_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            current_input_size = hidden_size

        layers.append(nn.Linear(current_input_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- 4. DATA PIPELINE ---

def get_dataloaders(file_path, batch_size, seed):
    try:
        data = pd.read_csv(file_path, header=None, sep=r'\s+')
    except FileNotFoundError:
        print(f"Error: File {file_path} no found!。")
        return None, None, None
        
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Label Shift (e.g., 1,2,3 -> 0,1,2)
    y = y - 1 

    x_all_tensor = torch.tensor(x, dtype=torch.float32)
    y_all_tensor = torch.tensor(y, dtype=torch.long)

    data_size = len(y_all_tensor)
    indices = torch.randperm(data_size)

    val_size = int(np.floor(0.1 * data_size))
    test_size = int(np.floor(0.1 * data_size))
    train_size = data_size - (test_size + val_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    x_train = x_all_tensor[train_indices]
    y_train = y_all_tensor[train_indices]
    x_val = x_all_tensor[val_indices]
    y_val = y_all_tensor[val_indices]
    x_test = x_all_tensor[test_indices]
    y_test = y_all_tensor[test_indices]

    # Normalization
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    epsilon = 1e-8 
    
    x_train = (x_train - mean) / (std + epsilon)
    x_val = (x_val - mean) / (std + epsilon)
    x_test = (x_test - mean) / (std + epsilon)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# --- 5. CORE FUNCTIONS ---

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, earlystop, device):
    # Store Convergence Curve
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    BEST_VAL_LOSS = float('inf')
    STOP_COUNTER = 0
    best_model_weights = None

    print("Start Trainning...")
    
    for epoch in range(num_epochs):
        # --- Trainning Mode ---
        model.train()
        epoch_train_losses = []
        epoch_train_correct = 0
        epoch_train_total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(x_batch)
            train_loss = criterion(outputs, y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Record
            epoch_train_losses.append(train_loss.item())
            _, predicted = torch.max(outputs.data, 1)
            epoch_train_total += y_batch.size(0)
            epoch_train_correct += (predicted == y_batch).sum().item()

        # --- Validation Mode ---
        model.eval()
        epoch_val_losses = []
        epoch_val_correct = 0
        epoch_val_total = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                val_outputs = model(x_batch)
                val_loss = criterion(val_outputs, y_batch)
                
                epoch_val_losses.append(val_loss.item())
                _, predicted = torch.max(val_outputs.data, 1)
                epoch_val_total += y_batch.size(0)
                epoch_val_correct += (predicted == y_batch).sum().item()

        # --- Store epoch reslut ---
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        train_acc = epoch_train_correct / epoch_train_total
        val_acc = epoch_val_correct / epoch_val_total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # # --- Check Early Stopping ---
        if avg_val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = avg_val_loss
            STOP_COUNTER = 0
            # Save Best Model Weight (Use deepcopy)
            best_model_weights = deepcopy(model.state_dict())
        else:
            STOP_COUNTER += 1

        if STOP_COUNTER >= earlystop:
            print(f"Early stopping at epoch {epoch+1}!")
            break
            
        if (epoch + 1) % 1000 == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    print("Trainning Success！")
    
    if best_model_weights:
        print(f"Already load best Model Weight, Lowest Validation Loss ={BEST_VAL_LOSS:.4f}")
        model.load_state_dict(best_model_weights)
        
    return model, history

def test_model(model, criterion, test_loader, device):
    model.eval()
    test_losses = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            test_outputs = model(x_batch)
            test_loss = criterion(test_outputs, y_batch)
            test_losses.append(test_loss.item())

            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += y_batch.size(0)
            test_correct += (test_predicted == y_batch).sum().item()

    avg_test_loss = np.mean(test_losses)
    test_acc = test_correct / test_total

    print(f'=======================================')
    print(f'Final Test Loss: {avg_test_loss:.4f}')
    print(f'Final Test Accuracy: {test_acc * 100:.2f}%')
    print(f'=======================================')

def plot_curves(history):

    plt.figure(figsize=(12, 5))

    # Plot Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Error')
    plt.plot(history['val_loss'], label='Validation Error')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Error)')
    plt.legend()

    # Plot Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# --- 6. MAIN EXECUTION ---

if __name__ == "__main__":
    
    # Environment Setting
    set_seed(SEED)
    print(f'Using device: {DEVICE}')

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        file_path=DATA_FILE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    if train_loader:
        model = MLP(input_size=INPUT_FEATURES, hidden_sizes=HIDDEN_LAYERS_CONFIG, output_size=OUTPUT_CLASSES, dropout_p=DROP).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()

        # 1. Gradint:
        # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        # 2. Momentum:
        # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=DECAY)

        # 3. Adam (常用)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
        # ---------------------

        # Model Tarinning
        model, history = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=MAX_ITERATION,
            earlystop=STOP,
            device=DEVICE
        )

        # Model Testing
        test_model(
            model=model,
            criterion=criterion,
            test_loader=test_loader,
            device=DEVICE
        )

        # 6. Plot result
        plot_curves(history)