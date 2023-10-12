from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:8080",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

#model = tf.keras.models.load_model("../saved_model/3")

endpoint = "http://localhost:8502/v1/models/tomato_model:predict"

class_names = ['Bacterial spot', 'Early blight', 'Late blight',
               'Late blight', 'Septoria leaf spot', 'Spider mites Two-spotted spider mite',
               'Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'healthy']

# read data
@app.get("/ali")
async def ali():
    return "Hello world, I am Ali Adel!, 23"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    # UploadFile is your data type
    file: UploadFile = File(...)
):
    # I need to convert that file into numpy array or tensor to make prediction
    # then read that file & await here for sending multi requestes images
    image = read_file_as_image(await file.read())
    # then I need to import image batch and make prediction
    img_batch = np.expand_dims(image, 0)
    # I will get from prediction a ndarray(1,10classes)
    #prediction = model.predict(img_batch)

    # json_data is the request
    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

    # then I need to get the highest value in array and that is the class
    # argmax here to return index of max value in the array
    #predicted_class = class_names[np.argmax(prediction[0])]
    #confidence = np.max(prediction[0])
    # Return a dictionary with class name and confidence percentage
    # return {
    #     'class': predicted_class,
    #     'confidence': float(confidence)
    # }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
