import cv2
import os
from insightface.app import FaceAnalysis

PERSON_NAME = "sample"

save_dir = f"face_db/{PERSON_NAME}"
os.makedirs(save_dir, exist_ok=True)

app = FaceAnalysis()
app.prepare(ctx_id=0)

cap = cv2.VideoCapture(0)

count = len(os.listdir(save_dir))

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Saved: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1)

    if key == ord("s"):

        if len(faces) > 0:

            face = faces[0]

            x1, y1, x2, y2 = face.bbox.astype(int)

            face_img = frame[y1:y2, x1:x2]

            count += 1

            filename = os.path.join(
                save_dir,
                f"{count}.jpg"
            )

            cv2.imwrite(filename, face_img)

            print("saved:", filename)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()