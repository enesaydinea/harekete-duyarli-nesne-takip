import cv2

cap = cv2.VideoCapture("ornek_video.mp4")

# Hareket algılamak için MOG2 algoritması
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hareketli yerleri algıla
    mask = fgbg.apply(frame)

    # Orijinal görüntü ve hareket maskesini göster
    cv2.imshow("Orijinal Video", frame)
    cv2.imshow("Hareket Maskesi", mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
