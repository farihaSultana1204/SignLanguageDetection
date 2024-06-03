import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from PIL import Image, ImageTk

# Initialize OpenCV capture and other variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

# Define labels and folder for data collection
labels = ["1", "2", "A", "B", "Hello"]
data_folder = "Data"

# Create a Tkinter window
window = tk.Tk()
window.title("Gesture Recognition App")

# Create a canvas for displaying the camera feed
canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()

def recognize_gesture():
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Display the camera feed in the Tkinter window
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.image = img

            # Display the predicted label on the window
            label_text.set("Predicted Gesture: " + labels[index])

        window.update()

# Create a button to start gesture recognition
start_button = tk.Button(window, text="Start Recognition", command=recognize_gesture)
start_button.pack()

# Create a label for displaying the predicted gesture
label_text = tk.StringVar()
label_text.set("Predicted Gesture: ")
gesture_label = tk.Label(window, textvariable=label_text)
gesture_label.pack()

window.mainloop()