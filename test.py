from learning_functions import Net
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

neural_net = Net().to(DEVICE)
neural_net.load_state_dict(torch.load('digit_identifier.pth'))
