import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

data_path = "dataset"

faces = []
ids = []

student_id = 1

for student_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, student_name)

    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            img = Image.open(image_path).convert('L')
            img_numpy = np.array(img, 'uint8')

            faces.append(img_numpy)
            ids.append(student_id)

        print(f"Training completed for {student_name}")

        student_id += 1

ids = np.array(ids)

recognizer.train(faces, ids)

recognizer.save("trainer.yml")

print("Model trained successfully!")