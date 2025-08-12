# utils/preprocess.py

from PIL import Image
import numpy as np

def preprocess_image(image_file, target_size=(224, 224)):
    """
    Preprocess uploaded image for prediction.
    - Resizes and normalizes the image.
    """
    image = Image.open(image_file).convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
