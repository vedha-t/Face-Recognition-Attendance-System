import cv2
import os

# Camera open
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Camera open aagala")
    exit()

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Student name
student_name = input("Enter student name: ")

# Dataset path
path = "dataset/" + student_name

# Folder create
if not os.path.exists(path):
    os.makedirs(path)

count = 0

while True:
    ret, img = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        count += 1

        # Save face image
        cv2.imwrite(
            f"{path}/{count}.jpg",
            gray[y:y+h, x:x+w]
        )

        # Rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

    cv2.imshow("Capturing Faces", img)

    # Stop after 30 images
    if count >= 30:
        break

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Face images captured successfully!")

cam.release()
cv2.destroyAllWindows()