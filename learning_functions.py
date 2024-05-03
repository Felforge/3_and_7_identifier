import torch.nn.functional as F
from torch import nn
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_FN = nn.CrossEntropyLoss()

class Net(nn.Module):
    """_summary_
    ML learning architecture
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50) # Fully connected layer
        self.fc2 = nn.Linear(50, 10) # Want to end with 10 for softmax
    def forward(self, x):
        """_summary_
        Activation for all things defined under __init__
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320) # 20 x 4 x 4
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)
