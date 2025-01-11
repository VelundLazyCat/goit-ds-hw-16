import streamlit as st
from PIL import Image
import time
from functions import is_image, image_predict


CNN = {"Classic CNN": 'cnn', "VGG-16": 'vgg16'}
# Заголовок сторінки
st.title(":blue[Застосунок класифікації зображеннь Fashion-MNIST]")

selectore = st.sidebar.selectbox('Select', ["Classic CNN", "VGG-16"])
st.sidebar.write(f"Обрано {selectore}")
st.sidebar.header(
    "Завантажте зображення для класифікації")
uploaded_file = st.sidebar.file_uploader(
    "Виберіть файл...", help='Прожмакай кнопку!', type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Відкриття зображення за допомогою Pillow
    try:
        image = Image.open(uploaded_file)
        # Відображення обробленого зображення
        st.sidebar.image(image, caption='Зображення для класифікації',
                         use_container_width=True)
    except:
        st.sidebar.write(f"завантажений файл не є зображенням")

    if is_image(uploaded_file):
        predicted_index, predicted_label, predicted_percent = image_predict(
            uploaded_file, CNN[selectore])
        st.write(f"Зображення класифіковане як {
                 predicted_label} з ймовірністю {predicted_percent} %")
        st.write(f"\n\n\nГрафіки точності та функціі втрат моделі {selectore}")
        image_cnn = Image.open(f'{CNN[selectore]}_loss_accuracy.png')
        # Відображення обробленого зображення
        st.image(image_cnn,  use_container_width=True)
