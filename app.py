from torchvision.transforms import ToTensor
from learning_functions import Net
from PIL import Image as im
import gradio as gr
import torch

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
        # Process input
        data = sketch_image['composite']
        data = im.fromarray(data)
        data = data.resize((28, 28))
        data = data.convert("LA")
        image_tensor = ToTensor()(data)
        image_tensor = image_tensor[1:,:,:].unsqueeze(0).to(DEVICE)

        # Get Predicition and Probabilities
        with torch.no_grad():
            output = model(image_tensor).sigmoid() - 0.5
        output_sum = torch.sum(output)
        probability_tensor = (output / output_sum) * 100
        probability_tensor = probability_tensor.to(torch.int32)
        probability_tensor = probability_tensor / 100
        probability_tensor = probability_tensor.to(torch.float32)
        return_labels = {}
        for i, elem in enumerate(probability_tensor):
            print(elem)
            num = elem.item()
            print(num)
            if num != 0.:
                return_labels[i] = num
        return return_labels
    return predict_inner

label = gr.Label()
sketchpad = gr.Sketchpad()

iface = gr.Interface(fn=predict(NEURAL_NET), inputs=sketchpad, outputs=label, title=TITLE)
iface.launch()
