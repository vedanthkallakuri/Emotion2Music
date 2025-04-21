import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("Cannot load haarcascade_frontalface_default.xml")


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return gray, faces


def detect_and_annotate(frame, model, emotion_dict):
    gray, faces = detect_faces(frame)
    out = frame.copy()
    emotion = None
    
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        crop = cv2.resize(roi, (48,48))
        crop = crop.astype('float32')/255.0
        crop = np.expand_dims(crop, axis=(0,-1))
        preds = model.predict(crop)
        idx = int(np.argmax(preds))
        emotion = emotion_dict[idx]

        cv2.rectangle(out, (x, y-50), (x+w, y+h+10), (255,0,0), 2)
        cv2.putText(out, emotion, (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if emotion is None:
        cv2.putText(out, "No face detected", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return out, emotion
