# Load the trained model
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("fall_detection_model.h5")
image_width=128
image_height=128
image_channels=3

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (image_width, image_height))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, image_height, image_width, image_channels))

    # Predict
    prediction = model.predict(reshaped_frame)
    label = 'Fall' if prediction > 0.5 else 'No Fall'

    # Display the result
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Fall Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
