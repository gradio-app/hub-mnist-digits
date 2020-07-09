import tensorflow as tf
import gradio as gr
from gradio.inputs import Sketchpad
from gradio.outputs import Label


mnist_model = tf.keras.models.load_model('models/mnist-99.0-acc.h5')

def predict(inp):
	prediction = mnist_model.predict(inp.reshape(1, 28, 28, 1)).tolist()[0]
	return {str(i): prediction[i] for i in range(10)}


sketchpad = Sketchpad()
label = Label(num_top_classes=4)

gr.Interface(
	predict, 
	sketchpad,  # could also be 'sketchpad' 
	label,
	capture_session=True,
	title="Real-time MNIST Sketchpad",
	description="See MNIST predictions realtime as you draw on a sketchpad.",
	thumbnail="https://raw.githubusercontent.com/gradio-app/real-time-mnist/master/thumbnail2.png",
	live=True,).launch();
