import torch
import torch.nn as nn

class Emg(nn.Module):
    def __init__(self, num_classes: int=2):
        super(Emg, self).__init__()
        
        #1° Blocco Convoluzionale
        #noi usiamo conv1D dato che i segnali sono unidimensionali, che variano solamente al variare del tempo
        #input [8, 200] -> dopo conv1 -> [16, 200] -> dopo il Pool -> [16, 100]
        self.c1=nn.Conv1d(in_channels=8, out_channels=16, kernel_size= 5, stride= 1, padding= 2)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool1d(kernel_size=2, stride=2)

        #2° Blocco Convoluzionale
        #input [16, 100] -> dopo conv2 -> [32, 100] -> dopo il Pool -> [32, 50]
        self.c2=nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.r2=nn.ReLU()
        self.p2=nn.MaxPool1d(kernel_size=2, stride=2)

        #3° Blocco Convoluzionale
        #input -> [32, 50] -> dopo conv3 -> [128, 1]
        self.c3=nn.Conv1d(in_channels=32, out_channels=128, kernel_size=50, stride=1, padding=0)
        self.r3=nn.ReLU()
        self.fc1= nn.Linear(128, num_classes)
        
    
    def forward(self, x):

        x= self.c1(x)
        x= self.r1(x)
        x= self.p1(x)

        x= self.c2(x)
        x= self.r2(x)
        x= self.p2(x)

        x=self.c3(x)
        x=self.r3(x)

        x=x.view(x.size(0), -1)
        x=self.fc1(x)

        return x
    
class EmgStandard(nn.Module):
    def __init__(self, num_classes: int=2):
        super(EmgStandard, self).__init__()

        #1° Blocco Convoluzionale
        #noi usiamo conv1D dato che i segnali sono unidimensionali, che variano solamente al variare del tempo
        #input [8, 200] -> dopo conv1 -> [16, 200] -> dopo il Pool -> [16, 100]
        self.c1=nn.Conv1d(in_channels=8, out_channels=16, kernel_size= 5, stride= 1, padding= 2)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool1d(kernel_size=2, stride=2)

        #2° Blocco Convoluzionale
        #input [16, 100] -> dopo conv2 -> [32, 100] -> dopo il Pool -> [32, 50]
        self.c2=nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.r2=nn.ReLU()
        self.p2=nn.MaxPool1d(kernel_size=2, stride=2)

        #Blocco lineare
        self.fc1= nn.Linear(32*50, num_classes)
        
    
    def forward(self, x):

        x= self.c1(x)
        x= self.r1(x)
        x= self.p1(x)

        x= self.c2(x)
        x= self.r2(x)
        x= self.p2(x)

        x=x.view(x.size(0), -1)
        x=self.fc1(x)

        return x



    
if __name__ == "__main__":
    model1=Emg()
    model2=EmgStandard()
    print (model1, model2)
