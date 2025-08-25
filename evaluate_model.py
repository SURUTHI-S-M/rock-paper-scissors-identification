import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# Parameters
img_height, img_width = 64, 64
batch_size = 16
num_classes = 3
epochs = 30

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data - only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training dataset
train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Load validation dataset
val_data = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Print class indices and sample data for debugging
print("Train class indices:", train_data.class_indices)
print("Validation class indices:", val_data.class_indices)

print("\n--- Sample train data filenames and labels ---")
for i in range(10):
    print(f"{train_data.filenames[i]} -> {train_data.labels[i]}")

print("\n--- Sample val data filenames and labels ---")
for i in range(10):
    print(f"{val_data.filenames[i]} -> {val_data.labels[i]}")


# Prepare class names list in order matching class_indices
class_names = [None] * num_classes
for name, idx in train_data.class_indices.items():
    class_names[idx] = name
print("\nClass names in order:", class_names)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)


sample_imgs, sample_labels = next(val_data)
prediction = model.predict(sample_imgs)  # sample_imgs shape: (batch_size, height, width, 3)

class_names = ['rock', 'paper', 'scissors']  # must match training order
predicted_index = np.argmax(prediction)
predicted_label = class_names[predicted_index]
print("Predicted gesture:", predicted_label)

print("\n--- Sample predictions ---")
for i in range(len(sample_imgs)):
    pred_class = np.argmax(prediction[i])
    true_class = np.argmax(sample_labels[i])
    print(f"Image {i+1}: Predicted: {class_names[pred_class]} (idx: {pred_class}), Actual: {class_names[true_class]} (idx: {true_class})")


# Save the trained model
model.save('gesture_cnn_model.h5')
print("\n✅ Model saved as gesture_cnn_model.h5")
