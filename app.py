from learning_functions import Net
import torch
import gradio as gr
from torchvision.transforms import ToTensor

# Look at this link and make my own API whenever
# https://github.com/fastai/tinypets/tree/master

TITLE = "MNIST Digit Identifier"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NEURAL_NET = Net().to(DEVICE)
NEURAL_NET.load_state_dict(torch.load('digit_identifier.pth'))
NEURAL_NET.eval()

def predict(model):
    """_summary_
    Returns prediciton from inputted model
    Args:
        model (pickle file): Learner class produced in notebook
    """
    def predict_inner(sketch_image):
        resized_image = sketch_image.resize((28,28))
        image_tensor = ToTensor()(resized_image)
        output = model(image_tensor.unsqueeze(0).to(DEVICE))
        prediction = output.argmax(dim=1, keepdim=True).item()
        return {prediction: 1.}
    return predict_inner

label = gr.Label()

iface = gr.Interface(fn=predict(NEURAL_NET), inputs="sketchpad", outputs=label, title=TITLE)
iface.launch()
