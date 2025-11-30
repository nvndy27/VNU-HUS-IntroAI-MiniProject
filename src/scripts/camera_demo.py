# camera_demo.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

IMG_SIZE = 224
MODEL_PATH = "model/model.h5"

# Load model
model = load_model(MODEL_PATH)
class_names = ['glass', 'metal', 'organic', 'paper', 'plastic', 'trash']

# Mở camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize + chuẩn hóa
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    label = class_names[class_idx]
    confidence = pred[0][class_idx]

    # Hiển thị kết quả
    cv2.putText(frame, f"{label}: {confidence*100:.1f}%", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Camera Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
