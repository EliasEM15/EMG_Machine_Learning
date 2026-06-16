


from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


class EmgDataloader(object):
    def __init__(self, data_folder_path: str, window_size: int=200, step_size: int=100): 
        #lista di informazioni che si useranno per il resto della classe
        #con window_size si intende la finestra temporale dove si svolgerà l'analisi del modello
        self.data_folder_path = data_folder_path
        self.window_size = window_size
        self.step_size = step_size
    
    def load_data(self):
        #x è il valore dei sensori, y è la verità assoluta (label)
        x_train, y_train = [], []
        x_test, y_test = [], []

  

        users = sorted([d for d in os.listdir(self.data_folder_path) if os.path.isdir(os.path.join(self.data_folder_path, d))])
        #la riga precedente serve a includere in un singolo vettore le cartelle di ogni individuo su cui è stata svolta l'analisi

        for user_id in users:
            user_folder = os.path.join(self.data_folder_path, user_id)

            for file in os.listdir(user_folder):
                #verifica dei file, affinché non vengano registrati file di altro tipo
                if not file.endswith('.txt'): continue
                file_path=os.path.join(user_folder, file)

                ds=pd.read_csv(file_path, sep='\t')

                #guardando i file.txt notiamo come la colonna finale è nominata class, parola riservata per python, la quale quindi potrebbe generare errori
                ds = ds.rename(columns={'class':'label'})
                
                # Definizioni delle classi selezionate per lo studio
                target_classes = [1,2,3,4,5,6,7]

                ds_filtered= ds[ds['label'].isin(target_classes)]
                ds_filtered['label']= ds_filtered['label'] - 1
                
                #separo i valori in due variabili: i risultato dei sensori(matrice di valori); i valori di label(verità assoluta)

                sensor_ds=ds_filtered.iloc[:, 1:9].values
                label_ds=ds_filtered['label'].values.astype(int)


                for i in range(0, len(sensor_ds) - self.window_size, self.step_size):

                    window= sensor_ds[i : i+self.window_size]
                    #siccome possono trovarsi all'interno della stessa finestra di lettura sia 
                    window_label=np.bincount(label_ds[i : i+ self.window_size]).argmax()

                    #separo il dataset che userò per il training con quello per il test
                    if int(user_id)<=24:
                        x_train.append(window)
                        y_train.append(window_label)

                    else:
                        x_test.append(window)
                        y_test.append(window_label)

        
        return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))
    

def preprocess_data(x_train, x_test):
    #come il Dott. Pimenta aveva normalizzato il valore dei pixel, si andrà quindi a normalizzare le tensioni in uscita dai nostri sensori
    mean= np.mean(x_train, axis=(0,1), keepdims=True)
    std= np.std(x_train, axis=(0,1), keepdims=True)
    
    #non possiamo normalizzare per la natura del segnale, il quale non ha un effettivo valore limite fisso, ma esso varia a seconda della persona considerata
    train_normalized= (x_train - mean) /(std + 1e-8)
    test_normalized= (x_test - mean) /(std + 1e-8)

    train_tensor = torch.tensor(train_normalized, dtype=torch.float32)
    test_tensor = torch.tensor(test_normalized, dtype=torch.float32)

    #analizzando la composizione dei tensori, notiamo che non sono nell'ordine prestabilito, dobbiamo quindi cambiare di posizione dle 
    train_tensor= train_tensor.transpose(1,2)
    test_tensor= test_tensor.transpose(1,2)

    return train_tensor, test_tensor


    





if __name__ == "__main__":
   
    base_path= Path(__file__).resolve().parent
    data_folder_path = base_path / "dataset" 
    

    estrattore = EmgDataloader(data_folder_path)
    
    (x_train, y_train), (x_test, y_test) = estrattore.load_data()
    train_tensor, test_tensor = preprocess_data(x_train, x_test)