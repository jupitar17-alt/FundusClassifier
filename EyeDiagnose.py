import cv2
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from random import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import shutil

# Define the path to the folder containing all the images
data_dir = 'C:/Users/jupit/Documents/Personal Project/AllEyeData'

# Define the path to the folder where the training and validation data will be saved
train_dir = 'C:/Users/jupit/Documents/Personal Project/train_data'
validation_dir = 'C:/Users/jupit/Documents/Personal Project/validation_data'

# Create the training and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Create subdirectories for each class (eye disease) in the training and validation directories
for class_name in os.listdir(data_dir):
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)

def apply_clahe(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(gray_img)
    equalized_img_rgb = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2RGB)
    return equalized_img_rgb

# Split the data into training and validation sets (80% training, 20% validation)
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    image_files = os.listdir(class_dir)
    np.random.shuffle(image_files)
    num_train = int(0.8 * len(image_files))
    train_files = image_files[:num_train]
    validation_files = image_files[num_train:]
    for file_name in train_files:
        src = os.path.join(class_dir, file_name)
        dst = os.path.join(train_dir, class_name, file_name)
        img = apply_clahe(src)
        cv2.imwrite(dst, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    for file_name in validation_files:
        src = os.path.join(class_dir, file_name)
        dst = os.path.join(validation_dir, class_name, file_name)
        img = apply_clahe(src)
        cv2.imwrite(dst, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Define image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255  # Rescale pixel values to [0, 1]
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
for layer in base_model.layers[:-10]:  # Adjust the number of layers to be trainable
    layer.trainable = False #Change to false if still not working
# Define and compile the model (same as before)
model = tf.keras.models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    BatchNormalization(),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

model.summary()
# Save the model
model.save('eye_disease_classifier_model.h5')
