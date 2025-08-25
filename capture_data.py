import cv2
import os

# Labels for Rock Paper Scissors
labels = ["rock", "paper", "scissors"]

# Directory to store captured images
DATA_DIR = "dataset"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of images per gesture
IMAGES_PER_LABEL = 100

cap = cv2.VideoCapture(0)

for label in labels:
    print(f"Collecting images for '{label}'...")

    # Create folder for the label
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    print("Press 's' to start capturing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to open camera.")
            break

        cv2.putText(frame, f"Ready to capture: {label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Capture Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    count = 0
    while count < IMAGES_PER_LABEL:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw rectangle around capture area
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

        # Show capturing info
        cv2.putText(frame, f"Capturing: {label} ({count+1}/{IMAGES_PER_LABEL})",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capture Data", frame)

        # Save cropped ROI
        roi = frame[100:400, 100:400]
        cv2.imwrite(os.path.join(label_path, f"{count}.jpg"), roi)

        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("Data collection complete!")
cap.release()
cv2.destroyAllWindows()
print("✅ Data captured and saved in the 'data' directory.")