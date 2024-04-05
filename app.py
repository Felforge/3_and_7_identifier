from fastai.vision.all import load_learner
import gradio as gr

# Look at this link and make my own API whenever
# https://github.com/fastai/tinypets/tree/master

LEARNER = load_learner("Akash_Food_or_Not.pkl")
TITLE = "Will Akash eat this food?"
EXAMPLES = ['example_pictures/butter_chicken.png', 'example_pictures/samosa.png',
            'example_pictures/burger.png']

def will_akash_eat(learn):
    """_summary_
    Take in learning model and image and return propbability
    of either possible outcome

    Args:
        learn (learning model): pickle file imported with load_learner from fastai
        image_name (image file): image file used in conjucntion with learn to make a predicition
    """
    def will_akash_eat_inner(image_name):
        # Original labels
        # labels = learn.dls.vocab
        labels = ["STEALING THIS OUT THE FRIDGE 😋🍴", "HEEEELLLL NAAAAHHH 🤮"]
        _, _, probs = learn.predict(image_name)
        return {labels[i]: float(probs[i]) for i in range(len(labels))}
    return will_akash_eat_inner

image = gr.Image()
label = gr.Label()

iface = gr.Interface(fn=will_akash_eat(LEARNER), inputs=image, outputs=label, examples=EXAMPLES, 
                     title=TITLE)
iface.launch()
