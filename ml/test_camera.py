import cv2


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("تعذر فتح الكاميرا.")

while True:
    success, frame = camera.read()

    if not success:
        print("تعذر قراءة الصورة من الكاميرا.")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()