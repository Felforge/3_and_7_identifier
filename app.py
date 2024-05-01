from learning_functions import Net
import torch
import gradio as gr

# Look at this link and make my own API whenever
# https://github.com/fastai/tinypets/tree/master

TITLE = "MNIST Digit Identifier"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NEURAL_NET = Net().to(DEVICE)
NEURAL_NET.load_state_dict(torch.load('digit_identifier.pth'))

def predict(model):
    """_summary_
    Returns prediciton from inputted model
    Args:
        model (pickle file): Learner class produced in notebook
    """
    def predict_inner(sketch_image):
        output = model(sketch_image)
        prediction = output.argmax(dim=1, keepdim=True).item()
        return prediction
    return predict_inner

label = gr.Label()

iface = gr.Interface(fn=predict(NEURAL_NET), inputs="sketchpad", outputs=label, title=TITLE)
iface.launch()
