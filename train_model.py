import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Count images in dataset folders
print("Train rock count:", len(os.listdir("dataset/train/rock")))
print("Train paper count:", len(os.listdir("dataset/train/paper")))
print("Train scissors count:", len(os.listdir("dataset/train/scissors")))
print("Val rock count:", len(os.listdir("dataset/val/rock")))
print("Val paper count:", len(os.listdir("dataset/val/paper")))
print("Val scissors count:", len(os.listdir("dataset/val/scissors")))

# Parameters
img_height = 64
img_width = 64
batch_size = 16

# Data generator with rescaling and validation split (if needed)
datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_data = datagen.flow_from_directory(
    'dataset/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load validation data
val_data = datagen.flow_from_directory(
    'dataset/val',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Print detected classes
print("Train classes:", train_data.class_indices)
print("Validation classes:", val_data.class_indices)

# Define the model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Save the model
model.save("model.h5")
print("✅ Model saved as model.h5")
import cv2
import numpy as np