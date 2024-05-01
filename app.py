from fastai.vision.all import load_learner
import gradio as gr

# Look at this link and make my own API whenever
# https://github.com/fastai/tinypets/tree/master

LEARNER = load_learner("digit_identifier.pkl")
TITLE = "MNIST Digit Identifier"

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

iface = gr.Interface(fn=predict(LEARNER), inputs="sketchpad", outputs=label, title=TITLE)
iface.launch()
