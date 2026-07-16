import cv2

# Face detector load pannudhu
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Camera open pannudhu
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera open aagala")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        print("Frame kidaikala")
        break

    # Image ah gray ah convert pannudhu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detect pannudhu
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Face ku box podum
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # Camera screen show pannum
    cv2.imshow("Face Detection", frame)

    # q press panna close aagum
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()