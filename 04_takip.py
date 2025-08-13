# 04_takip.py
import cv2
import numpy as np
from pathlib import Path
from centroidtracker import CentroidTracker

# === Mutlak yol ile video ===
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "ornek_video.mp4"   # video dosyası bu klasörde olmalı

cap = cv2.VideoCapture(str(VIDEO_PATH))
ct = CentroidTracker()
fgbg = cv2.createBackgroundSubtractorMOG2()

cv2.namedWindow("Takip")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # kare gelmiyorsa video bitmiş/okunamıyor demektir

    # MOG2 ile hareket maske çıkarımı + basit temizlik
    mask = fgbg.apply(frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Konturlar -> dikdörtgenler
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        if cv2.contourArea(c) < 800:  # küçük paraziti ele
            continue
        x, y, w, h = cv2.boundingRect(c)
        rects.append((x, y, x + w, y + h))

    # Centroid listesi
    inputs = [((sx + ex) // 2, (sy + ey) // 2) for (sx, sy, ex, ey) in rects]
    inputs = np.array(inputs) if inputs else np.empty((0, 2), dtype=int)

    # Takip güncelle
    objects = ct.update(inputs)

    # Çizimler (sırayla eşle)
    for ((sx, sy, ex, ey), objID) in zip(rects, list(objects.keys())):
        cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {objID}", (sx, sy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Göster
    cv2.imshow("Takip", frame)

    # 🔑 Pencere olaylarını işle (gerekli yoksa pencere gri kalır)
    cv2.waitKey(1)

    # X'e basıldıysa çık
    if cv2.getWindowProperty("Takip", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
ws()
