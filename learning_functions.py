import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

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
        
def batch_accuracy(xb, yb):
    """Return accuracy of predicitions as an integer in a tensor

    Args:
        xb (torch.tensor): tensor of predicitions
        yb (torch.tensor): tensor of labels to match images with 

    Returns:
        torch.tensor: Accuracy of predicitions as a tensor
    """
    correct = xb == yb
    return correct.float().mean()

def train(model, train_dl, epoch):
    model.train()
    global optimizer
    for batch_idx, (data, target) in enumerate(train_dl):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = LOSS_FN(output, target)
        loss.backward()
        optimizer = optimizer.step()
        if batch_idx % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_dl.dataset)} ({100. * batch_idx / len(train_dl):.0f}%)]\t{loss.item():.6f}")
        
def test(model, valid_dl):
    model.eval()
    test_loss = 0
    correct = 0
    global optimizer

    with torch.no_grad():
        for data, target in valid_dl:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += LOSS_FN(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valid_dl.dataset)
    print(f'\nTest Loss: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{len(valid_dl.dataset)} {100. * correct / len(valid_dl.dataset):.0f}%')
