import torch
from torch.nn import Module


class FC_Model(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FC_Model, self).__init__()
        
        self.f1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)

        self.f2 = torch.nn.Linear(hidden_size, hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)

        self.f3 = torch.nn.Linear(hidden_size,int(hidden_size/4))
        self.bn3 = torch.nn.BatchNorm1d(int(hidden_size/4))

        self.fout = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.f1(x)
        out = self.relu(out)
        out = self.bn1(out)
        ''' section commented out since it does not help to improve accuracy
        out = self.f2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.f3(out)
        out = self.relu(out)
        out = self.bn3(out)
        '''
        out = self.fout(out)
        out = self.sigmoid(out)
        return out



