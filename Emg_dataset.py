



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
        x_train, y_train = [], []
        x_test, y_test = [], []

        # Definizioni delle classi selezionate per lo studio
        target_classes = [1, 2]

        users = sorted([d for d in os.listdir(self.data_folder_path) if os.path.isdir(os.path.join(self.data_folder_path, d))])
        #la riga precedente serve a includere in un singolo vettore le cartelle di ogni individuo su cui è stata svolta l'analisi

        for user_id in users:
            user_folder = os.path.join(self.data_folder_path, user_id)

            for file in os.listdir(user_folder):
                #verifica dei file, affinché non vengano registrati file di altro tipo
                if not file.endswith('.txt'): continue
                file_path=os.path.join(user_folder, file)

        
    