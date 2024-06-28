import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np


# Initialize the main window
window = tk.Tk()
window.title("Face Recognition Model")
window.geometry("900x500")
window.configure(bg="#2c3e50")

# Define styles
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 16), background="#2c3e50", foreground="#ecf0f1")
style.configure("TButton", font=("Helvetica", 16), background="#3498db", foreground="#ecf0f1")
style.configure("TEntry", font=("Helvetica", 14))

# Function to train the classifier
def train_classifier():
    data_dir = "C:/Users/1mahe/Desktop/Face Recognition/data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result', 'Training Dataset Completed!')

# Function to detect faces
def detect_face():
    def draw_boundary(img, classifier, scaleFactor, minN, color, text, clf):
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = classifier.detectMultiScale(g_img, scaleFactor, minN)

        coords = []

        for (x, y, w, h) in feats:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(g_img[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 77:
                if id == 1:
                    cv2.putText(img, "Maher", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                if id == 2:
                    cv2.putText(img, "Chaker", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                if id == 3:
                    cv2.putText(img, "Yassin", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            
            else:
                cv2.putText(img, "CHKOUN HADHA?", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img = cv2.flip(img,1)
        img = recognize(img, clf, faceCascade)
        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) == 13:
            break
    video_capture.release()
    cv2.destroyAllWindows()

# Function to generate dataset
def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo('Result', 'Please fill all the user infos!')
    else:
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_crop(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]
            return cropped_face

        cap = cv2.VideoCapture(0)
        id = 1
        img_id = 0

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
            
            if face_crop(frame) is not None:
                img_id += 1
                face = cv2.resize(face_crop(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = "data/user." + str(id) + "." + str(img_id) + ".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped Face", face)
                if cv2.waitKey(1) == 13 or int(img_id) == 200:
                    break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Collecting Samples is Completed ... DONE')

# Create buttons with hover effects
def on_enter(e):
    e.widget['background'] = '#2980b9'

def on_leave(e):
    e.widget['background'] = '#3498db'

b1 = tk.Button(window, text="Training", font=("Helvetica", 16), bg="#3498db", fg="#ecf0f1", command=train_classifier)
b1.grid(column=0, row=1, padx=10, pady=20, sticky="nsew")
b1.bind("<Enter>", on_enter)
b1.bind("<Leave>", on_leave)

b2 = tk.Button(window, text="Detect the face", font=("Helvetica", 16), bg="#3498db", fg="#ecf0f1", command=detect_face)
b2.grid(column=1, row=1, padx=10, pady=20, sticky="nsew")
b2.bind("<Enter>", on_enter)
b2.bind("<Leave>", on_leave)

b3 = tk.Button(window, text="Generate Dataset", font=("Helvetica", 16), bg="#3498db", fg="#ecf0f1", command=generate_dataset)
b3.grid(column=2, row=1, padx=10, pady=20, sticky="nsew")
b3.bind("<Enter>", on_enter)
b3.bind("<Leave>", on_leave)

    # Header
header = ttk.Label(window, text="Face Recognition Model", font=("Helvetica", 24, "bold"))
header.grid(column=0, row=0, columnspan=3, pady=20, sticky="nsew")

# Updating a label with changing text
changing_label = ttk.Label(window, text="", font=("Helvetica", 16))
changing_label.grid(column=0, row=2, columnspan=3, pady=20, sticky="nsew")

def update_label_text():
    messages = ["Processing...", "Please wait...", "Almost there...", "Done!"]
    current_index = 0

    def change_text():
        nonlocal current_index
        changing_label.config(text=messages[current_index])
        current_index = (current_index + 1) % len(messages)
        window.after(1000, change_text)
    change_text()

# Start label text update
    update_label_text()

# Centering all widgets in the window    zina :)
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)
window.grid_columnconfigure(2, weight=1)
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)
window.grid_rowconfigure(2, weight=1)
header = ttk.Label(window, text="Face Recognition Model", font=("Helvetica", 24, "bold"), anchor=tk.CENTER)
header.grid(column=0, row=0, columnspan=3, pady=20, sticky="nsew")


window.mainloop()
