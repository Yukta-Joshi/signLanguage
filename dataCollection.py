import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time

# Open the webcam
cap = cv2.VideoCapture(1)

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Offset for cropping around hand landmarks
offset = 40
# Size of the cropped and resized images
imgSize = 300
# Counter for saving images
counter = 0
# Folder to save images
# folder = "Data/Hello"
folder = "DataBank/Withdarw"

while True:
    # Read frame from the webcam
    success, img = cap.read()

    # Check if the frame is successfully read
    if not success:
        print("Error: Failed to read frame from the webcam.")
        break

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
            x_min, y_min, x_max, y_max = 9999, 9999, 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)

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
                    if aspect_ratio > 1:
                        k = imgSize / imgCrop.shape[0]
                        w_cal = math.ceil(k * imgCrop.shape[1])
                        img_resize = cv2.resize(imgCrop, (w_cal, imgSize))
                        w_gap = math.ceil((imgSize - w_cal) / 2)
                        imgWhite[:, w_gap:w_cal + w_gap] = img_resize
                    else:
                        k = imgSize / imgCrop.shape[1]
                        h_cal = math.ceil(k * imgCrop.shape[0])
                        img_resize = cv2.resize(imgCrop, (imgSize, h_cal))
                        h_gap = math.ceil((imgSize - h_cal) / 2)
                        imgWhite[h_gap:h_cal + h_gap, :] = img_resize

            else:
                print("Hand landmarks not detected")

            # Display the cropped and resized images
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image Resized", imgWhite)

    # Display the original image with hand landmarks
    cv2.imshow("Image", img)
    # Wait for a key press
    key = cv2.waitKey(1)

    # Press 'Esc' to exit the loop
    if key == 27:
        break

    # Press 's' to save the current frame as an image
    if key == ord("s"):
        counter += 1
        timestamp = int(time.time())
        image_name = f'Image_{timestamp}_{counter}.jpg'
        image_path = os.path.join(folder, image_name)

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save the image
        saved = cv2.imwrite(image_path, imgWhite)
        if saved:
            print(f"Image {counter} ({image_name}) saved at {image_path}")
        else:
            print(f"Failed to save image {counter} ({image_name})")

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
