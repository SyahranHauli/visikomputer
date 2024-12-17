import os
import kaggle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# Fungsi untuk mengunduh dataset dari Kaggle
def download_dataset():
    # Nama dataset yang akan diunduh
    dataset_name = 'hari31416/food-101'
    download_path = 'food-101'
    
    # Mengunduh dataset
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    return os.path.abspath(download_path)

# Mengunduh dataset (jika belum diunduh)
dataset_path = download_dataset()
st.write(f"Dataset berhasil diunduh di path: {dataset_path}")

# Data preprocessing
batch_size = 32
img_size = (128, 128)  # Ukuran gambar yang lebih kecil untuk mempercepat proses

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# Build model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model (optional: load model if already trained)
epochs = 100
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# Simpan model
model.save('food_model.h5')

# Load the model for inference
model = tf.keras.models.load_model('food_model.h5')

# Food Info Dictionary
food_info = {
    'macarons': {'calories': 250, 'healthy': 'No'},
    'sashimi': {'calories': 200, 'healthy': 'Yes'},
    'beet_salad': {'calories': 150, 'healthy': 'Yes'},
    'bibimbap': {'calories': 600, 'healthy': 'Yes'},
    'ceviche': {'calories': 180, 'healthy': 'Yes'},
    'garlic_bread': {'calories': 350, 'healthy': 'No'},
    'oysters': {'calories': 50, 'healthy': 'Yes'},
    'risotto': {'calories': 400, 'healthy': 'No'},
    'seaweed salad': {'calories': 60, 'healthy': 'Yes'},
    'steak': {'calories': 700, 'healthy': 'No'}
}

# Streamlit UI
st.title('Food Image Classification')
st.write("Upload an image to predict its food category and nutritional information.")

uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_resized = img.resize(img_size)
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_name = list(train_data.class_indices.keys())[class_idx]

    # Display prediction
    info = food_info.get(class_name, {'calories': 'Unknown', 'healthy': 'Unknown'})

    # Show image and prediction
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: {class_name}")
    st.write(f"Calories: {info['calories']} kcal")
    st.write(f"Healthy: {info['healthy']}")

    # Plotting the image and prediction result
    st.subheader("Prediction Result Visualization")
    fig, ax = plt.subplots()
    ax.imshow(img_resized)
    ax.axis('off')
    ax.set_title(f"{class_name}\nKalori: {info['calories']} kcal, Sehat: {info['healthy']}")
    st.pyplot(fig)

    # Optionally display confusion matrix and classification report
    if st.checkbox('Show Confusion Matrix'):
        y_pred = model.predict(val_data)
        y_true = val_data.labels
        cm = confusion_matrix(y_true, y_pred.argmax(axis=1))

        fig_cm = plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.class_indices.keys(), yticklabels=train_data.class_indices.keys())
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        plt.title('Confusion Matrix')
        st.pyplot(fig_cm)
        
    if st.checkbox('Show Classification Report'):
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred.argmax(axis=1), target_names=val_data.class_indices.keys())
        st.text(report)
