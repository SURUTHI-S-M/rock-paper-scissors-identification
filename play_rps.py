import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("model.h5")

# Parameters matching your model input
img_height, img_width = 64, 64

# Open webcam
cap = cv2.VideoCapture(0)

class_names = ['paper', 'rock', 'scissors']  # Make sure this order matches your model classes!

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    img = cv2.resize(roi, (img_width, img_height))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_id = np.argmax(predictions[0])
    # confidence = predictions[0][class_id]  # No longer needed for display

    text = f"{class_names[class_id]}"  # Only show gesture name

    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Game started. Press 'q' to exit.")
