import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('gesture_cnn_model.h5')

# Class names must match the order used during training
class_names = ['rock', 'paper', 'scissors']

# Parameters matching your training setup
img_height, img_width = 64, 64

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess frame for model: resize, convert color, scale pixels
    img = cv2.resize(frame, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # Predict gesture
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_label = class_names[predicted_class_index]

    # Display predicted label on frame
    cv2.putText(frame, f'Gesture: {predicted_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Rock Paper Scissors - Gesture Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Live gesture recognition started. Press 'q' to exit.")