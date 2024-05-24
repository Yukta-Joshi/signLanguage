import streamlit as st
import cv2
from cvzone.ClassificationModule import Classifier
import mediapipe as mp
import numpy as np
import math
import time
from streamlit_option_menu import option_menu
import  streamlit_toggle as tog
from gtts import gTTS
import pygame
import io
#For sentence forming
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize Pygame mixer for audio playback
pygame.mixer.init()

custom_css = """
    <style>
    [data-testid="stApp"]{
   
        background-color: #EBE9E0; /* Set background color */
        width: 100%; /* Make the Streamlit app container occupy the full width */
        padding: 0; 
        margin: 0;
        }
    
    [data-testid="stHeader"]{
   
        background-color: #51829B; /* Set background color */
        width: 100%; /* Make the Streamlit app container occupy the full width */
        padding: 20; 
        margin: 0;
        }
        
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Return preprocessed text as a single string
    return ' '.join(tokens)


def generate_sentence(predicted_text):
    # Preprocess the predicted text
    processed_text = predicted_text.capitalize() + "."  # Capitalize first letter and add period
    return processed_text


def guide():
    st.write("A step-by-step guide on how to use the system.")
    Gcol1, Gcol2 = st.columns(2)

    with Gcol1:
        st.image("step-1.png", width=300)
        st.write("Step 1: Go to website")

        st.image("step3.png", width=300)
        st.write("Step 3: Let your client be 2 feet apart and then make gestures.")

    with Gcol2:
        st.image("step-2.png", width=300)
        st.write("Step 2: Click cam button from the side menu")

        st.image("step-4.png", width=300)
        st.write("Step 4: Look at the recognized gestures.")

# Initialize voiceover status
voiceover_enabled = True

def toggle_voiceover():
    global voiceover_enabled
    voiceover_enabled = not voiceover_enabled
    if voiceover_enabled:
        st.write("Voiceover enabled")
    else:
        st.write("Voiceover disabled")

def speak_text(text):
    # Only generate audio output if voiceover is enabled
    if voiceover_enabled:
        # Use gTTS to generate speech
        tts = gTTS(text)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        # Use pygame to play the audio from BytesIO
        pygame.mixer.music.load(fp, 'mp3')
        pygame.mixer.music.play()

def camera():
    # st.markdown("# Sign Language Recognition")
    previous_predicted_text = ""
    predicted_text = ""
    predicted_sentence = " "
    predicted_labels = []
    # Open webcam
    cap = cv2.VideoCapture(0)
    # if st.button("Open Webcam"):
    if not cap.isOpened():
        st.warning("Failed to open webcam.")
        return

    # Create an empty space for video display
    video_placeholder = st.empty()


    # Initialize Mediapipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2)
    mp_drawing = mp.solutions.drawing_utils

    classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")

    # Offset for cropping around hand landmarks
    offset = 20
    # Size of the cropped and resized images
    imgSize = 300
    # Counter for saving images
    counter = 0

    labels = [label.strip() for label in open("Models/labels.txt", "r").readlines()]

    # Threshold for prediction confidence
    confidence_threshold = 0.95

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
                cv2.rectangle(img, (x_min - offset, y_min - offset), (x_max + offset, y_max + offset), (0, 0, 0), 2)

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
                        if aspect_ratio > 1:  # image is taller than it is wide
                            k = imgSize / imgCrop.shape[0]
                            w_cal = math.ceil(k * imgCrop.shape[1])
                            img_resize = cv2.resize(imgCrop, (w_cal, imgSize))
                            w_gap = math.ceil((imgSize - w_cal) / 2)
                            imgWhite[:, w_gap:w_cal + w_gap] = img_resize

                        else:  # image is wider than it is hight
                            k = imgSize / imgCrop.shape[1]
                            h_cal = math.ceil(k * imgCrop.shape[0])
                            img_resize = cv2.resize(imgCrop, (imgSize, h_cal))
                            h_gap = math.ceil((imgSize - h_cal) / 2)
                            imgWhite[h_gap:h_cal + h_gap, :] = img_resize

                    time.sleep(0.2)

                    # make predictions
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print("prediction: ", prediction, "\nindex: ", index)
                    confidence_score = prediction[index]

                    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
                    confidence = str(int(confidence_score * 100)) + '%'

                    if confidence_score >= confidence_threshold and 0 <= index < len(labels):
                        # Display the predicted label and confidence on the image
                        label_text = f"{labels[index]} ({confidence})"
                        print(label_text)

                        cv2.rectangle(img, (x_min - offset, y_min - offset - 50),
                                      (x_min - offset + 300, y_min - offset - 40 + 40), (0, 0, 0), cv2.FILLED) #fill small rectangle
                        # cv2.putText(img, labels[index], (x_min, y_min - 46), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(img, label_text, (x_min, y_min - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                        # Get the predicted text
                        predicted_text = labels[index]

                        # Append predicted text to the list
                        predicted_labels.append(predicted_text)
                        # Generate the sentence
                        predicted_sentence = generate_sentence(predicted_text)




        # clear_button_counter = 0

        # Convert frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Update the video placeholder with the new frame
        video_placeholder.image(frame_rgb, channels="RGB")

        if predicted_text != previous_predicted_text:
            st.sidebar.write(" ", predicted_sentence)
            st.sidebar.write("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
            previous_predicted_text = predicted_text

            # Speak the predicted text
            speak_text(predicted_sentence)

            time.sleep(0.1)

# Create a button to toggle voiceover
on = st.toggle("Voiceover", value=True, key="voiceover_toggle")


if on:
    toggle_voiceover()



    # # Release the webcam
    # cap.release()

def main():

    # st.title("Sign Language Recognition")
    st.markdown("<h1 style='text-align: center; color: black;'>Sign Language Recognition</h1>", unsafe_allow_html=True)


    # Create a button
    clicked = option_menu(
        menu_title=None,
        options=["Guide", "Camera"],
        icons=["book", "camera"],
        orientation="horizontal"
    )

    if clicked == "Camera":
        camera()
    elif clicked == "Guide":
        guide()


if __name__ == "__main__":
    main()
