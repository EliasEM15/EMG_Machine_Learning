### step 4 - Evaluate the Model (Identico a Pimenta)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from time import time
from pathlib import Path

# Importiamo i pezzi che abbiamo costruito negli altri tuoi due file
from Emg_dataset import EmgDataloader, preprocess_data
from Emg_model import Emg


BATCH_SIZE = 64
EPOCHS = 15          
LEARNING_RATE = 0.001

#caricamento dei dati
base_path= Path(__file__).resolve().parent
data_folder_path = base_path / "dataset" 
estrattore= EmgDataloader(data_folder_path="./dataset")
(x_train, y_train), (x_test, y_test) = estrattore.load_data()

#preparo i tensori usati per training e test
_, x_test = preprocess_data(x_train, x_test)
#non ho bisogno di caricare le y del training in questo caso
y_test= torch.tensor(y_test, dtype=torch.long)


#preparazione dei dataLoaders. Bisogna fare un passaggio ulteriore dato che in uscita a preprocess_data non ho un unico dataset
test_ds= TensorDataset(x_test, y_test)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


#setup del training
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=Emg(num_classes=2).to(device)
model_filepath = base_path / "emg_lenet1d.pth"
model.load_state_dict(torch.load(model_filepath))
model.eval()

# 3. Disattiviamo il calcolo dei gradienti per risparmiare memoria e velocizzare i calcoli
with torch.no_grad():
    correct = 0
    total = 0
    
    # Ciclo sui dati del Test Set (gli utenti rimasti fuori dall'addestramento)
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Passaggio in avanti
        outputs = model(inputs)
        
        # Troviamo il gesto con il punteggio più alto (0 o 1)
        _, predicted = torch.max(outputs.data, 1)
        
        # Accumuliamo il totale delle finestre e quelle indovinate
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Stampa finale della precisione percentuale, speculare a Pimenta
    print(f"Accuracy of the network on the test dataset: {100 * correct / total:.2f} %")