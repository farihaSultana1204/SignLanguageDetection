import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = None
offset = 20
imgSize = 300
labels = ["1", "2", "A", "B", "Hello"]


data_folder = "Data"
counter = 0

root = tk.Tk()
root.title("Sign Language Detection")


notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)


style = ttk.Style()
style.configure("Welcome.TFrame", background="Teal")


welcome_frame = ttk.Frame(notebook, style="Welcome.TFrame")
notebook.add(welcome_frame, text="Welcome")

welcome_messages = ["Welcome to the Sign Language Detection App",
                    "Start Detecting Signs Now!"]
welcome_label = ttk.Label(welcome_frame, text="", font=("Lobster", 20), background="Gold")
welcome_label.pack(pady=(150, 10))


current_message_index = 0

def animate_welcome():
    global current_message_index
    welcome_label.config(text=welcome_messages[current_message_index])
    current_message_index = (current_message_index + 1) % len(welcome_messages)
    root.after(2000, animate_welcome)

animate_welcome()


detection_frame = ttk.Frame(notebook)
notebook.add(detection_frame, text="Detection")


canvas = tk.Canvas(detection_frame, width=640, height=480)
canvas.pack()


label_var = tk.StringVar()
sign_label = ttk.Label(detection_frame, textvariable=label_var, font=("Helvetica", 20))
sign_label.pack()


def update():
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands, _ = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = frame[y-offset:y + h+offset, x-offset:x + w+offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        label_var.set(labels[index])

    frame = cv2.resize(frame, (640, 480))
    photo = PhotoImage(data=cv2.imencode('.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes())
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    canvas.photo = photo
    canvas.after(10, update)


def start_detection():
    notebook.select(1)
    canvas.pack()
    update()


start_button = ttk.Button(welcome_frame, text="Start Detection", command=start_detection)
start_button.pack(pady=(0, 10))


try:
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
except Exception as e:
    print(f"Error loading the model: {e}")

root.mainloop()
