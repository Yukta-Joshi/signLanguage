import cv2
from cvzone.ClassificationModule import Classifier
import mediapipe as mp
import numpy as np
import math
import os
import time

# Open the webcam
cap = cv2.VideoCapture(0)


# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

classifier = Classifier("Model-v2/keras_model.h5","Model-v2/labels.txt")

# Offset for cropping around hand landmarks
offset = 20
# Size of the cropped and resized images
imgSize = 300
# Counter for saving images
counter = 0

labels = [label.strip() for label in open("Model-v2/labels.txt", "r").readlines()]

# Threshold for prediction confidence
confidence_threshold = 0.75

while True:
    # Read frame from the webcam
    success, img = cap.read()

    # Check if the frame is successfully read
    if not success:
        print("Error: Failed to read frame from the webcam.")
        break

    # No mirror effect
    # img = cv2.flip(img, 1)

    # Convert image to RGB format
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process hand landmarks using Mediapipe hands module
    results = hands.process(rgb_img)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on the image
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Find bounding box around hand landmarks
            # Initialize variables to store bounding box coordinates
            x_min, y_min, x_max, y_max = 9999, 9999, 0, 0

            for landmark in hand_landmarks.landmark:
                # Get the pixel coordinates of the landmark
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                # Update bounding box coordinates
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
            # Draw bounding box rectangle around the detected hand region
            cv2.rectangle(img, (x_min - offset, y_min - offset), (x_max + offset, y_max + offset), (0, 0, 225), 2)

            # Crop the image around hand landmarks
            imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]

            # Create a white background image
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:  # Check if the cropped image has non-zero dimensions
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCropShape = imgCrop.shape
                aspect_ratio = imgCropShape[0] / imgCropShape[1]

                # Check if the width of the cropped image is non-zero to avoid division by zero
                if imgCropShape[1] > 0:

                    # Resize and center the cropped image on the white background
                    if aspect_ratio > 1: #image is taller than it is wide
                        k = imgSize / imgCrop.shape[0]
                        w_cal = math.ceil(k * imgCrop.shape[1])
                        img_resize = cv2.resize(imgCrop, (imgSize, imgSize))
                        w_gap = math.ceil((imgSize - w_cal) / 2)
                        imgWhite[:, w_gap:w_cal + w_gap] = img_resize

                    else: #image is wider than it is hight
                        k = imgSize / imgCrop.shape[1]
                        h_cal = math.ceil(k * imgCrop.shape[0])
                        img_resize = cv2.resize(imgCrop, (imgSize, imgSize))
                        h_gap = math.ceil((imgSize - h_cal) / 2)
                        imgWhite[h_gap:h_cal + h_gap, :] = img_resize

                    # Debug print statement to check the shape of the resized image
                    print("Resized image shape:", img_resize.shape)
                time.sleep(0.05)
                #make predictions
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print("prediction: ", prediction, "\nindex: ", index)
                confidence_score = prediction[index]

                print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

                if confidence_score >= confidence_threshold and 0 <= index < len(labels):
                    # Display the predicted label and confidence on the image
                    label_text = f"{labels[index]} ({prediction[0]:.2f})"

                    cv2.rectangle(img, (x_min - offset, y_min - offset - 50),
                                  (x_min - offset + 200, y_min - offset - 40 + 30), (0, 0, 0),
                                  cv2.FILLED)  # fill small rectangle
                    cv2.putText(img, labels[index], (x_min, y_min - 46), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                                2)


            else:
                print("Hand landmarks not detected")



            # Display the cropped and resized images
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image Resized", imgWhite)

        # cv2.putText(img)

    # Display the original image with hand landmarks
    cv2.imshow("Image", img)
    # Wait for a key press
    cv2.waitKey(1)


# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


