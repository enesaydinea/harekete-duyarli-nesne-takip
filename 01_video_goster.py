import cv2

# Video dosyasının yolu
video_yolu = "ornek_video.mp4"
cap = cv2.VideoCapture(video_yolu)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video", frame)

    # 'q' tuşuna basarsan çık
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
