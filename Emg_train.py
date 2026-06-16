from pathlib import Path
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from Emg_dataset import EmgDataloader, preprocess_data
from Emg_model import Emg, EmgStandard, EmgMultiScale


#def train_model():
BATCH_SIZE = 64      
EPOCHS = 40       
LEARNING_RATE = 0.001
NUM_CLASSES= 7

#caricamento dei dati
base_path= Path(__file__).resolve().parent
data_folder_path = base_path / "dataset" 
estrattore= EmgDataloader(data_folder_path=data_folder_path)
(x_train, y_train), (x_test, y_test) = estrattore.load_data()

#preparo i tensori usati per training e test
x_train, x_test = preprocess_data(x_train, x_test)
#i tensori delle y (le verità assolute), non erano ancora stati eseguiti dato che bastava una sola riga di comando
y_train= torch.tensor(y_train, dtype=torch.long)
y_test= torch.tensor(y_test, dtype=torch.long)

#preparazione dei dataLoaders. Bisogna fare un passaggio ulteriore dato che in uscita a preprocess_data non ho un unico dataset
train_ds= TensorDataset(x_train, y_train)
test_ds= TensorDataset(x_test, y_test)

class_counts= torch.bincount(y_train)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


#setup del training
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=EmgMultiScale(num_classes=NUM_CLASSES).to(device)
print(f"Dispositivo utilizzato per l'allenamento: {model.__class__.__name__}")
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

training_start_time= time()

#training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_start_time = time()
    #usiamo r_loss per calcolare l'errore per ogni epoca in modo da capire se l'errore diminuisce
    r_loss=0.0
    #usiamo correct_train e total_train per calcolare l'accuratezza del modello. Se l'accuratezza è troppo alta probabilmente sta andando in overfitting
    correct_train=0.0
    total_train=0.0

    for inputs, labels in train_loader:
        inputs, labels= inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss= loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        r_loss += loss.item()

        #a parte il calcolo della loss si aggiunge il calcolo dell'accuratezza per ogni epoca, in modo da verificare eventuali overfittimg
        _, predicted=torch.max(outputs.data, 1)
        total_train+= labels.size(0)
        correct_train+= (predicted==labels).sum()

    epoch_end_time=time()
    epoch_time = epoch_end_time - epoch_start_time

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            t_loss = loss_function(outputs, labels)
            test_loss += t_loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    scheduler.step(avg_test_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}],\t Loss: {r_loss/len(train_loader):.4f},\t Acc: {(correct_train / total_train) * 100:.4f},\t elapsed time: {epoch_time:.4f} seconds")

training_end_time=time()
training_time= training_end_time - training_start_time
print(f"Training completed in {training_time:.4f} seconds")

#salvo i pesi ricavati finora
model_filepath = base_path / "emg_lenet1d.pth"
torch.save(model.state_dict(), model_filepath)
print(f"Model saved to {model_filepath}")