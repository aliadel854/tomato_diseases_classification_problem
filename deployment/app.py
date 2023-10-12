import cv2
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)


model = tf.keras.models.load_model("../saved_model/3")

class_names = ['Bacterial spot', 'Early blight', 'Late blight',
               'Late blight', 'Septoria leaf spot', 'Spider mites Two-spotted spider mite',
               'Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'healthy']

@app.route('/')
def index():
	return render_template("index.html", data="hey")


@app.route("/prediction", methods=["POST"])
def prediction():
	img = request.files['img']

	img.save("img.jpg")

	image = cv2.imread("img.jpg")

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = cv2.resize(image, (256, 256))

	image = np.reshape(image, (1, 256, 256, 3))

	pred = model.predict(image)
	# then I need to get the highest value in array and that is the class
	# argmax here to return index of max value in the array
	predicted_class = class_names[np.argmax(pred)]

	confidence = np.max(pred)

	prediction = {
	    'Class_Name': predicted_class,
	    'Confidence': float(confidence),
	}

	# Return a dictionary with class name and confidence percentage
	return render_template('prediction.html', data=prediction)

if __name__ == "__main__":
	app.run(debug=True)