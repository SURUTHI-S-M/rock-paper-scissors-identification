import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
labels = ["rock", "paper", "scissors"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = labels[np.argmax(prediction)]

    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Game started. Press 'q' to exit.")
