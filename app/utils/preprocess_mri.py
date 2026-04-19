import numpy as np
from PIL import Image
import io


def load_and_resize_mri_image(file_bytes: bytes, size=(224, 224)):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize(size)
    return image


def preprocess_mri_image(file_bytes: bytes):
    image = load_and_resize_mri_image(file_bytes)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_mri_pil_image(image: Image.Image):
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array