import logging

FORMAT = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
# Use filename="file.log" as a param to logging to log to a file
logging.basicConfig(format=FORMAT, level=logging.INFO)


"""# Inferencing"""

import os
import logging
import numpy as np

class Model:
    def __init__(self, source_path='/content'):
        self.source_path = source_path 
        self.model = None
        self.load_model()
    def load_model(self, filename='keras_model.h5'):
        from tensorflow.keras.models import load_model
        self.model = load_model(f'{self.source_path}/{filename}') 
        return self.model
    def predict(self, img):
         # Note: Image here has already been preprocessed.
         import numpy as np
         model_pred = np.argmax(self.model.predict(img).ravel())
         model_prob = round(np.max(self.model.predict(img).ravel()), 3)
         values = ["adapter", "bearing", "gear"]
         prediction = values[model_pred]
         final_pred = {"adapter": "Adapter Pk2", "bearing": "Ratchet Clamp Ring Stainless Steel", "gear": "Miter Gear-10 Pitch"}
         result = final_pred[prediction]
         result = f'"{result}"'
         model_prob2 = f'"{model_prob}"'
         final_str = '{"Prediction": ' + result + ', ' + '"Probability": ' + model_prob2 + '}'
         return final_str
    
class ImageProcess:
    def __init__(self, source_path='/content'):
        self.source_path = source_path
        self.img = None
    
    def load_img_from_bytes(self, image_data): 
        from PIL import Image
        import io
        self.img = Image.open(io.BytesIO(image_data)) 
        return self.preprocess_img()
    
    def preprocess_img(self):
        import numpy as np
        from PIL import Image,ImageOps
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224,224)
        img_array = np.asarray(
            ImageOps.fit(self.img, size, Image.ANTIALIAS))
        normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_img_array
        return data
