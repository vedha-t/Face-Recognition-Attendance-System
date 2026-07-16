import cv2
import mysql.connector
from datetime import datetime

# ---------------- MySQL Connection ----------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Vedha@2006",   # Unga MySQL password
    database="face_attendance"
)

cursor = conn.cursor()

# ---------------- Face Recognizer ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Student Names
names = ["Unknown", "vedha"]

# Duplicate attendance avoid panna
attendance_marked = set()

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        student_id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        print("ID:", student_id, "Confidence:", confidence)

        if confidence < 95:

            name = names[student_id]

            # Attendance oru thadava mattum save aagum
            if name not in attendance_marked:

                now = datetime.now()

                current_date = now.strftime("%Y-%m-%d")
                current_time = now.strftime("%H:%M:%S")

                sql = """
                INSERT INTO attendance(student_name, date, time)
                VALUES (%s, %s, %s)
                """

                values = (name, current_date, current_time)

                cursor.execute(sql, values)
                conn.commit()

                attendance_marked.add(name)

                print(f"{name} Attendance Marked")

        else:
            name = "Unknown"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

cursor.close()
conn.close()