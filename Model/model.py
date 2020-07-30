import torch.nn as nn

class OneDimCNN(nn.Module):
    def __init__(self):
        super(OneDimCNN, self).__init__()
        self.num_class = 34

        self.conv = nn.Conv2d(1, 32, kernel_size=(1, 71), stride=(1, 1))
        self.pool = nn.MaxPool2d(1, 2)
        self.hidden1 = nn.Linear(50*32*1, 128)
        self.hidden2 = nn.Linear(128, self.num_class)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.hidden1(x))
        x = self.softmax(self.hidden2(x))

        return x