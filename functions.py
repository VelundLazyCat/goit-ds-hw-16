import numpy as np
from PIL import Image
import os
import tensorflow as tf


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height


def is_image(file_path):
    try:
        with Image.open(file_path) as img:
            return True
    except:
        return False


def preprocess_image_vgg(image: str):
    raw_img = tf.io.read_file(image)
    img = tf.io.decode_image(raw_img, dtype=tf.float32, channels=3)
    img = tf.image.resize_with_crop_or_pad(img, 32, 32)
    image_array = img / 255.0
    input_image = np.expand_dims(image_array, axis=0)
    return input_image


def preprocess_image_vgg_alt(image: str):
    test_image = Image.open(image).convert("RGB")
    target_size = (32, 32)
    test_image = test_image.resize(target_size)
    image_array = np.array(test_image)
    image_array = image_array / 255.0
    input_image = np.expand_dims(image_array, axis=0)
    return input_image


def preprocess_image_cnn(image: str):
    test_image = Image.open(image)
    target_size = (28, 28)
    test_image = test_image.resize(target_size)
    image_array = np.array(test_image)
    image_array = image_array / 255.0
    input_image = np.expand_dims(image_array, axis=0)
    return input_image


def image_predict(image: str, cnn: str) -> tuple:
    mnist_classes = ['Футболка/топ', 'Штани', 'Светр', 'Сукня',
                     'Пальто', 'Сандалі', 'Сорочка', 'Кросівки', 'Сумка', 'Чоботи']

    loaded_model = tf.keras.models.load_model(f"trained_model_{cnn}.keras")

    # Preprocess the image for prediction
    if cnn == 'cnn':
        processed_image = preprocess_image_cnn(image)
    elif cnn == 'vgg16':
        processed_image = preprocess_image_vgg_alt(image)

    # Make prediction using the loaded model
    predictions = loaded_model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = mnist_classes[predicted_class_index]
    predicted_class_percent = round(
        predictions[0][predicted_class_index]*100, 2)

    return predicted_class_index, predicted_class_label, predicted_class_percent


if __name__ == "__main__":
    print(image_predict('img/10.png', 'cnn'))
    print(image_predict('img/10.png', 'vgg16'))
