# -*- coding: utf-8 -*-
"""
Inference script that extends from the base infer interface
"""
import os
from os.path import exists
from joblib import load
import logging

from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from tf_template import Model, ImageProcess

app = Flask(__name__)
CORS(app, support_credentials=True)
img_process = None
model = None



@app.before_first_request 
def init():
    """
    Load the model if it is available locally
    """
    print('in init()')
    import tensorflow as tf
    import logging
    logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    global img_process, model
    
    
    img_process = ImageProcess() 
    
    
    model = Model(source_path=os.environ['SERVE_FILES_PATH']) 

    return None


@app.route("/v1/predict", methods=["POST"])
@cross_origin(supports_credentials=True)
def predict():
    """
    Perform an inference on the model created in initialize

    Returns:
        String prediction of the label for the given test data
    """
    global model, img_process
    payload = dict(request.json) 
    import base64
    img_data = base64.b64decode(payload['imgData'])
    prediction = model.predict(img_process.load_img_from_bytes(img_data))
    
    return prediction, 200, {"Content-Type":"application/json"}
                             

if __name__ == "__main__":
    logging.info("SERVE_FILES_PATH=", os.environ['SERVE_FILES_PATH'])
    init()
    app.run(host="0.0.0.0", debug=True, port=9001) #to start the server

# curl --location --request POST 'http://localhost:9001/v1/predict' --header 'Content-Type: application/json' --data-raw '{"text": "A restaurant with great ambiance"}'
